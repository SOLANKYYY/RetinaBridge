# RetinaBridge — ONNX migration test

This separate test app runs your exported model using Python and ONNX Runtime, without MATLAB installed on the web server. It includes the uploaded model and the existing patient intake/report interface adapted from SOLANKYYY/RetinaBridge.

**Status: executable migration candidate, not a validated replacement for the MATLAB app.**

The ONNX graph passed the ONNX checker and runs on CPU. Its input is float32 NCHW `[batch,3,224,224]`, in RGB 0–255 units. Z-score normalization is embedded in the graph and must not be applied twice. The output is five softmax scores. Classes 0–4 follow the original model screenshot; the comparison utility verifies this ordering against MATLAB.

Model SHA-256: `4c068e7289376f56c72d34ac6d5c56c905ae12be23c850b2fbdf36748af819de`

## Important remaining differences

- The MATLAB enhancement function was not exported with the network. `inference.py` contains a candidate OpenCV/Pillow port. MATLAB and Python resizing, Lab conversion, CLAHE, smoothing, rounding, and quality calculations can differ. No original accuracy or sensitivity number can be claimed for this pipeline yet.
- Grad-CAM and vessel candidates are now included. See the explanation implementation below. Their output has not been compared with MATLAB on fundus images.
- Heuristic quality checks can accept non-fundus images. The synthetic test intentionally demonstrates this limitation.
- Patient details remain in browser memory; image bytes are sent to the Python server. No database or saved image history is included.

## First: run this separate folder locally

Do not overwrite your existing MATLAB project. Extract this archive to Downloads. Open the extracted `RetinaBridge-ONNX` folder in VS Code, then open Terminal → New Terminal.

Use Python 3.12. On Windows, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py
```

If the Python launcher is unavailable but `python --version` reports 3.12, use `python -m venv .venv` for the first command.

Open http://127.0.0.1:5000. Stop your old server first if it is already using port 5000. Keep the terminal open. Use authorised demonstration images only. A prominent notice identifies the migration limitations. Ctrl+C stops the app.

The model loads once at startup. A load error prevents startup rather than returning fake results. The development server binds only to localhost.

## Next: compare with your MATLAB model

Do this before treating the predictions as equivalent or announcing deployment readiness.

1. In MATLAB, set Current Folder to this extracted app's `matlab` directory. It contains a copy of the repository's original `preprocess_fundus.m` and a new helper.
2. Run:

```matlab
export_validation_sample
```

3. Choose the ORIGINAL project's active `dr_model.mat`, then an authorised fundus test image, then an output directory. The helper creates a uniquely named sample folder containing `original.png`, `enhanced.png`, and `reference.json`. It does not train or modify the original model.
4. In a second PowerShell terminal in this app directory, run the following with the printed sample path:

```powershell
.\.venv\Scripts\python.exe compare_sample.py "C:\path\to\sample-folder"
```

5. Share the printed comparison output. Share sample images only if you are authorised to do so.

The utility separately compares:

- MATLAB vs ONNX scores using the EXACT MATLAB-enhanced pixels: tests model export, normalization, tensor layout, and class order.
- MATLAB pipeline vs Python pipeline on the original image: measures preprocessing drift, score differences, and quality acceptance changes.

`export_close` uses absolute and relative tolerances of 1e-4 as an engineering check, not a clinical criterion. One matching grade does not validate a migration. Repeat across grades, sizes, and accepted/rejected images; resolve preprocessing differences and evaluate the candidate on the original held-out dataset. Retain original model metrics separately.

## Later: password-protected Render demo

This configuration is for a restricted engineering demonstration after reviewing the comparison. Hosting does not validate the pipeline. Grad-CAM and vessel candidate views are included; MATLAB parity remains unverified.

1. Put this directory's contents in a separate GitHub repository or a separate development branch. Keep `models/dr_model.onnx`. Use Git for the approximately 44.8 MB model; do not attempt GitHub's browser uploader for this file. Do not commit real patient images or validation fixtures.
2. On Render choose **New → Web Service** and connect that repository/branch.
3. Set the root directory to the folder containing this `server.py` and `requirements.txt` (blank if these are at repository root).
4. Set runtime to Python, build command to `pip install -r requirements.txt`, and start command to:

```sh
python -m gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

5. Set health-check path to `/healthz`. Set `DEMO_PASSWORD` to a long random password. The app refuses startup on Render without it.
6. Set `PUBLIC_ORIGIN` to the EXACT assigned HTTPS origin, for example `https://your-service.onrender.com`, with no trailing slash or path. Update it if using a custom domain. This is required for browser uploads behind the HTTPS proxy.
7. Use Python 3.12 (the included `.python-version` requests it). Choose an instance after checking current pricing and measured memory usage; no free-tier fit or uptime guarantee is asserted here.
8. Deploy, then sign in with username `demo` and your `DEMO_PASSWORD`. Both frontend and inference run on the same service. No MATLAB or paid AI API key is required.

Use one worker initially: the concurrency lock is process-local. This demo has no account management, durable audit system, or distributed queue. Basic authentication must be used over HTTPS when hosted. `/healthz` is intentionally public and discloses only readiness/runtime.

See [Render Flask deployment](https://render.com/docs/deploy-flask) and [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).

## Verification performed

- Uploaded ONNX model passed `onnx.checker.check_model` and loaded with CPUExecutionProvider.
- Startup inference and synthetic request inference returned finite five-class softmax outputs.
- Ten unittest cases passed, including explanation checks described below: model/frontend availability, black-image rejection without grade, synthetic execution, invalid/small/oversized/origin-rejected uploads, busy response, and demo authentication.
- Python syntax compilation and JavaScript syntax checks passed.

Run the checks locally with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_backend test_explanations -v
```

Not performed: MATLAB-vs-ONNX comparison (needs original MATLAB reference outputs), clinical/held-out evaluation, browser visual inspection, Windows installation, or actual Render deployment. The source UI layout was preserved apart from migration labels, server wording, and a visible limitation notice. The code and tests are included for review.

## Explanation implementation (updated build)

The uploaded model has a final 512x7x7 feature tensor (`res5b_relu`), global average pooling and a linear five-class head. For class logit z_c, its derivative with respect to feature A_kij is exactly W_kc/49. The implementation averages these analytical gradients, combines the actual model features, applies ReLU, and upsamples the result. This is logit Grad-CAM for this architecture, equivalent to normalized CAM here; it does not require retraining, PyTorch, a GPU or a separate saliency model. It is not a generic Grad-CAM implementation for arbitrary ONNX models.

The model hash is checked at startup. Exposing the feature tensor occurs only in memory; the original ONNX file is unchanged. Every prediction verifies that the feature tensor and classifier weights reconstruct the returned softmax scores. A blank positive map remains uncoloured. Rejected inputs receive no class heatmap. Coarse 7x7 attention cannot establish lesion boundaries.

Vessel candidates use the enhanced green channel, disk-radius-5 morphological black-hat filtering, a local threshold, and exclusion of dark background. This is an approximate port of the original heuristic, not a trained vessel model and not MATLAB pixel parity. It can include noise and other dark structures. A uniform image returns an empty mask.

Tests cover unchanged classification scores versus the untouched model, analytical gradients versus finite differences, class-specific maps, rejection behavior and binary vessel masks with known synthetic lines. Synthetic fixtures test implementation, not anatomical accuracy. No real fundus image or MATLAB attention reference was supplied in this turn.

Method reference: [Grad-CAM paper](https://arxiv.org/abs/1610.02391).
