"""Usage: python compare_sample.py PATH_TO_SAMPLE_FOLDER

Two separate comparisons: export on identical pixels, and preprocessing drift.
A matching class on one image is not clinical validation or pipeline parity.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from inference import Predictor, preprocess

folder = Path(sys.argv[1])
ref = json.loads((folder / 'reference.json').read_text())
if [str(x) for x in ref['classes']] != ['0', '1', '2', '3', '4']:
    raise ValueError('Class ordering does not match this backend.')
original = np.array(Image.open(folder / 'original.png').convert('RGB'))
matlab_image = np.array(Image.open(folder / 'enhanced.png').convert('RGB'))
candidate, quality = preprocess(original)
engine = Predictor()
matlab_scores = np.asarray(ref['scores']).reshape(5)
export_scores = engine.scores(matlab_image)
python_scores = engine.scores(candidate)
out = dict(
    matlab_scores=matlab_scores.tolist(), onnx_same_pixels=export_scores.tolist(),
    python_pipeline_scores=python_scores.tolist(),
    export_max_absolute_error=float(np.abs(export_scores-matlab_scores).max()),
    export_close=bool(np.allclose(export_scores, matlab_scores, atol=1e-4, rtol=1e-4)),
    preprocessing_mean_pixel_error=float(np.abs(candidate.astype(float)-matlab_image).mean()),
    full_pipeline_score_error=float(np.abs(python_scores-matlab_scores).max()),
    matlab_quality=ref['quality'], python_quality=quality,
    quality_decision_matches=quality['accepted']==ref['quality']['accepted'],
    matlab_grade=int(np.argmax(matlab_scores)), python_grade=int(np.argmax(python_scores)))
print(json.dumps(out, indent=2))
