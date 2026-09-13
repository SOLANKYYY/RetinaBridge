"""Flask ONNX migration test service. No MATLAB or external AI service required."""
import hmac
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from inference import Predictor, analyze

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
LOCK = threading.Lock()
predictor = Predictor()
PASSWORD = os.environ.get('DEMO_PASSWORD', '')
PUBLIC_ORIGIN = os.environ.get('PUBLIC_ORIGIN', '').rstrip('/')
if os.environ.get('RENDER') and not PASSWORD:
    raise RuntimeError('Set DEMO_PASSWORD before deploying the demonstration.')

@app.before_request
def access():
    if request.path == '/healthz':
        return None
    if PASSWORD:
        auth = request.authorization
        if not auth or auth.username != 'demo' or not hmac.compare_digest(auth.password or '', PASSWORD):
            return ('Demo access requires a password.', 401,
                    {'WWW-Authenticate': 'Basic realm="RetinaBridge demo"'})
    if request.method == 'POST':
        # Same-origin browser requests only; Basic auth is not a CSRF defense.
        origin = request.headers.get('Origin')
        expected = PUBLIC_ORIGIN or request.host_url.rstrip('/')
        if origin != expected:
            return jsonify(error='Open the screening page on this server and retry.'), 403

@app.after_request
def headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response

@app.get('/healthz')
def health():
    return jsonify(ready=True, runtime='onnxruntime')

@app.get('/api/status')
def status():
    return jsonify(model=True, runtime='ONNX Runtime CPU', migration_validated=False)

@app.get('/')
def index():
    return send_from_directory(ROOT / 'web', 'index.html')

@app.get('/<name>')
def asset(name):
    if name not in ('app.js', 'style.css'):
        return jsonify(error='Not found'), 404
    return send_from_directory(ROOT / 'web', name)

@app.post('/api/screen')
def screen():
    if not LOCK.acquire(blocking=False):
        return jsonify(error='Another scan is running. Please retry shortly.'), 429
    try:
        raw = request.get_data()
        if not raw:
            return jsonify(error='Choose an image first.'), 400
        return jsonify(analyze(raw, predictor))
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        return jsonify(error=str(exc)), 400
    finally:
        LOCK.release()

@app.errorhandler(413)
def large(_):
    return jsonify(error='Choose an image under 15 MB.'), 413

@app.errorhandler(Exception)
def failure(exc):
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    app.logger.exception('Screening request failed')
    return jsonify(error='Processing failed. Check the server log.'), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
