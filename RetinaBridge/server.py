"""Loopback-only MATLAB screening bridge. No external AI service."""
import base64, io, json, os, shutil, subprocess, tempfile, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image, UnidentifiedImageError
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock()
Image.MAX_IMAGE_PIXELS=25_000_000

def matlab_exe():
    value=os.environ.get('MATLAB_EXE','matlab')
    return shutil.which(value) or (value if Path(value).is_file() else None)

def analyze(raw):
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in ('PNG','JPEG'): raise ValueError('Choose a PNG or JPEG fundus image.')
        if min(im.size)<224: raise ValueError('Image is too small. Use an image at least 224 × 224 pixels.')
        im.load(); rgb=im.convert('RGB')
    if not matlab_exe(): raise RuntimeError('MATLAB was not found. Set MATLAB_EXE as explained in START_HERE.md.')
    if not LOCK.acquire(blocking=False): raise RuntimeError('Another image is being processed. Please wait.')
    try:
        with tempfile.TemporaryDirectory(prefix='retina_') as d:
            d=Path(d); rgb.save(d/'input.png')
            env=os.environ.copy(); env['DR_JOB_DIR']=str(d); env['DR_PROJECT_DIR']=str(ROOT)
            command=[matlab_exe()]
            if os.name=='nt': command+=['-wait']
            command+=['-sd',str(ROOT/'matlab'),'-batch','screen_image']
            run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
            if run.returncode or not (d/'result.json').exists():
                print(run.stdout[-4000:],run.stderr[-2000:])
                raise RuntimeError('MATLAB could not finish. Check the server window for toolbox or model errors.')
            result=json.loads((d/'result.json').read_text())
            for name in ['enhanced','heatmap','vessels']:
                f=d/(name+'.png')
                if f.exists(): result[name]='data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()
            return result
    finally: LOCK.release()

class Handler(BaseHTTPRequestHandler):
    def send(self,status,data,kind='application/json'):
        body=json.dumps(data,allow_nan=False).encode() if kind=='application/json' else data
        self.send_response(status); self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path=='/api/status':
            return self.send(200,{'matlab':bool(matlab_exe()),'model':(ROOT/'models/dr_model.mat').exists()})
        allowed={'/':'index.html','/style.css':'style.css','/app.js':'app.js'}
        if self.path not in allowed:return self.send(404,{'error':'Not found'})
        f=ROOT/'web'/allowed[self.path]
        self.send(200,f.read_bytes(),{'html':'text/html; charset=utf-8','css':'text/css','js':'text/javascript'}[f.suffix[1:]])
    def do_POST(self):
        if self.path!='/api/screen':return self.send(404,{'error':'Not found'})
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:5000','http://localhost:5000'):
            return self.send(403,{'error':'Use the local website.'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=15*1024*1024:return self.send(413,{'error':'Choose an image under 15 MB.'})
            self.send(200,analyze(self.rfile.read(size)))
        except (ValueError,UnidentifiedImageError,Image.DecompressionBombError) as e:self.send(400,{'error':str(e)})
        except subprocess.TimeoutExpired:self.send(504,{'error':'MATLAB exceeded 5 minutes. Check MATLAB and try again.'})
        except RuntimeError as e:self.send(503,{'error':str(e)})
        except Exception:
            import traceback; traceback.print_exc(); self.send(500,{'error':'Processing failed. Check the server window.'})
if __name__=='__main__':
    print('RetinaBridge: http://127.0.0.1:5000 — Ctrl+C to stop')
    ThreadingHTTPServer(('127.0.0.1',5000),Handler).serve_forever()
