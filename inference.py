"""ONNX inference candidate. Python enhancement is NOT yet MATLAB-equivalent."""
import base64
import io
import os
import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import onnx
from PIL import Image

MODEL = Path(__file__).resolve().parent / 'models' / 'dr_model.onnx'
LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
Image.MAX_IMAGE_PIXELS = 25_000_000
cv2.setNumThreads(1)

class Predictor:
    def __init__(self, path=MODEL):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        # This analytic Grad-CAM is specific to the verified GAP -> linear head.
        # Refuse a different model until its graph has been checked explicitly.
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != '4c068e7289376f56c72d34ac6d5c56c905ae12be23c850b2fbdf36748af819de':
            raise ValueError('Model changed: verify the Grad-CAM head before using this build.')
        graph = onnx.load(str(path))
        tensors = {t.name: onnx.numpy_helper.to_array(t) for t in graph.graph.initializer}
        self.weights = tensors['dr_fc_MatMul_W'].copy()
        self.bias = tensors['dr_fc_Add_B'].copy()
        # Expose the final feature tensor without changing weights or predictions.
        graph.graph.output.append(onnx.helper.make_tensor_value_info(
            'res5b_relu', onnx.TensorProto.FLOAT, ['BatchSize', 512, 7, 7]))
        self.session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape[1:] != [3, 224, 224]:
            raise ValueError('Unexpected model input: expected NCHW RGB 224x224.')
        self.name = inputs[0].name
        # Startup smoke check exercises the runtime, not model accuracy.
        self.scores(np.zeros((224, 224, 3), dtype=np.uint8))

    def scores(self, enhanced):
        return self.forward(enhanced)[0]

    def forward(self, enhanced):
        if enhanced.shape != (224, 224, 3) or enhanced.dtype != np.uint8:
            raise ValueError('Expected an enhanced uint8 RGB image of size 224x224.')
        # Mean and standard deviation are already embedded in the ONNX graph.
        tensor = np.ascontiguousarray(enhanced.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.session.run(['prob', 'res5b_relu'], {self.name: tensor})
        values = np.asarray(outputs[0]).reshape(-1)
        features = outputs[1][0]
        if values.shape != (5,) or not np.isfinite(values).all():
            raise RuntimeError('Invalid model output.')
        if np.min(values) < -1e-6 or not np.isclose(values.sum(), 1, atol=1e-4):
            raise RuntimeError('Expected five softmax scores.')
        if features.shape != (512, 7, 7) or not np.isfinite(features).all():
            raise RuntimeError('Invalid Grad-CAM feature tensor.')
        logits = features.mean(axis=(1, 2)) @ self.weights + self.bias
        reconstructed = np.exp(logits - logits.max())
        reconstructed /= reconstructed.sum()
        if not np.allclose(values, reconstructed, atol=1e-5, rtol=1e-4):
            raise RuntimeError('Grad-CAM head does not match model scores.')
        return values, features

    def attention(self, enhanced):
        values, features = self.forward(enhanced)
        grade = int(np.argmax(values))
        heat = gradcam_map(features, self.weights, grade)
        overlay = enhanced.copy()
        if heat.max() > 0:
            colors = cv2.cvtColor(cv2.applyColorMap(
                np.rint(heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # Zero attention leaves the image unchanged; avoid a false blue blanket.
            alpha = (0.45 * heat)[..., None]
            overlay = np.rint((1-alpha)*enhanced + alpha*colors).astype(np.uint8)
        return values, overlay, bool(heat.max() > 0)

def gradcam_map(features, weights, grade):
    """Exact analytic logit Grad-CAM for this GAP + linear classification head.

    z_c = sum_k W[k,c] mean_ij(A[k,i,j]) + b_c.
    Therefore d(z_c)/d(A[k,i,j]) = W[k,c]/(H*W).
    Spatially averaging those gradients gives the same weights. Apply the
    Grad-CAM ReLU and normalize for display. No fabricated saliency is used.
    This target is the pre-softmax logit, not calibrated disease probability.
    """
    gradients = weights[:, grade] / (features.shape[1]*features.shape[2])
    raw = np.maximum(np.einsum('k,kij->ij', gradients, features), 0)
    if raw.max() <= 1e-12:
        return np.zeros((224, 224), np.float32)
    return np.clip(cv2.resize(raw/raw.max(), (224, 224), interpolation=cv2.INTER_LINEAR), 0, 1)

def vessel_candidates(enhanced):
    """Experimental black-hat green-channel mask, not trained segmentation.

    Follows the original MATLAB method family; thresholding is an OpenCV
    approximation, not a claim of pixel equivalence to imbinarize.
    """
    green = enhanced[:, :, 1]
    yy, xx = np.ogrid[-5:6, -5:6]
    disk = ((xx*xx + yy*yy) <= 25).astype(np.uint8)
    response = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, disk,
                                borderType=cv2.BORDER_REPLICATE)
    local = cv2.GaussianBlur(response.astype(np.float32), (31, 31), 0)
    mask = (response > np.maximum(local, 2)) & (green > 0.06*255)
    return mask.astype(np.uint8) * 255

def decode(raw):
    with Image.open(io.BytesIO(raw)) as im:
        if im.format not in ('PNG', 'JPEG'):
            raise ValueError('Choose a PNG or JPEG image.')
        if min(im.size) < 224 or im.width * im.height > 25_000_000:
            raise ValueError('Use an image at least 224x224 and at most 25 million pixels.')
        im.load()
        return np.array(im.convert('RGB'))

def preprocess(rgb):
    """Approximate port; resize/CLAHE/Lab rounding differ from MATLAB.

    The old model's validation metrics do not apply to this pipeline until
    paired checks and evaluation are completed. Thresholds remain experimental.
    """
    gray = (rgb.astype(np.float32) @ np.array([0.298936, 0.587043, 0.114021], np.float32)) / 255
    mask = gray > 0.04
    kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]], np.float32) / 6
    lap = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    coverage = float(mask.mean())
    brightness = float(gray[mask].mean()) if mask.any() else 0.
    focus = float(lap[mask].var(ddof=1)) if mask.sum() > 1 else 0.
    accepted = coverage > .25 and .08 < brightness < .92 and focus > .00003
    quality = dict(coverage=coverage, brightness=brightness, focus=focus, accepted=accepted,
                   feedback=('Experimental Python quality checks passed; confirm fundus identity.'
                             if accepted else 'Recapture: check focus, lighting and retinal coverage.'))
    # Pillow bicubic downsampling includes filtering, but is not MATLAB imresize.
    resized = np.array(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BICUBIC))
    lab = cv2.cvtColor(resized.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    light = np.rint(np.clip(lab[:, :, 0] / 100, 0, 1) * 255).astype(np.uint8)
    # OpenCV's clipLimit units differ from MATLAB's. This is a candidate only.
    light = cv2.createCLAHE(clipLimit=2.56, tileGridSize=(8, 8)).apply(light)
    lab[:, :, 0] = cv2.GaussianBlur(light.astype(np.float32) / 255, (3, 3), .4,
                                  borderType=cv2.BORDER_REPLICATE) * 100
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced = np.rint(np.clip(out, 0, 1) * 255).astype(np.uint8)
    return enhanced, quality

def png_data(rgb):
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode('ascii')

def analyze(raw, predictor):
    enhanced, quality = preprocess(decode(raw))
    result = dict(quality=quality, grade=None, label='', score=None,
                  enhanced=png_data(enhanced), heatmap=None,
                  vessels=png_data(vessel_candidates(enhanced)),
                  status='Image rejected — recapture required',
                  note='ONNX migration test only. Python enhancement is not yet validated against MATLAB. '
                       'Grad-CAM explains the predicted class logit at res5b_relu; it is not lesion evidence. '
                       'Vessel candidates are experimental green-channel filtering, not validated segmentation.')
    if quality['accepted']:
        scores, overlay, positive = predictor.attention(enhanced)
        grade = int(np.argmax(scores))
        result.update(grade=grade, label=LABELS[grade], score=float(scores[grade]),
                      heatmap=png_data(overlay),
                      attention_method='Analytic logit Grad-CAM',
                      status='Experimental prediction — pending review')
        if not positive:
            result['note'] += ' No positive class attention was found; the attention view is uncoloured.'
    return result
