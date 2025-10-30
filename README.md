

# AI-Powered Brain Tumor Detection System

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.0-orange)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/tensorflow-%3E%3D2.0-orange)](https://www.tensorflow.org)

An easy-to-run Streamlit app that loads a trained Keras model to detect and classify brain tumors from MRI scans. This repository focuses on inference (not training) and provides a clean UI for uploading MRI images and visualizing model predictions.

---

## 🔍 Features

- Fast local inference via Streamlit UI
- Binary or multi-class tumor detection (depends on your model)
- Confidence scores displayed for predictions
- Aspect-ratio preserving preprocessing for uploaded images
# AI-Powered Brain Tumor Detection System

An easy-to-run Streamlit application that loads a trained Keras model to detect and classify brain tumors from MRI scans. This repository contains the inference app (`app.py`) and expects a Keras model file (`brain_tumor_detection_model.h5`) in the project root.

## 🧠 Overview

This project provides a simple web interface to upload MRI images and receive model predictions (binary tumor detection or multi-class tumor types). It is intended for research and educational use only and is **not** a substitute for professional medical advice.

## ✨ Features

- Upload MRI scans (JPG/PNG) and get instant model predictions.
- Multi-class support for models trained to classify different tumor types (e.g., Glioma, Meningioma, Pituitary).

<!-- Project README for a polished GitHub appearance -->

# AI-Powered Brain Tumor Detection System

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.0-orange)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/tensorflow-%3E%3D2.0-orange)](https://www.tensorflow.org)

An easy-to-run Streamlit app that loads a trained Keras model to detect and classify brain tumors from MRI scans. This repository focuses on inference (not training) and provides a clean UI for uploading MRI images and visualizing model predictions.

---

## 🔍 Features

- Fast local inference via Streamlit UI
- Binary or multi-class tumor detection (depends on your model)
- Confidence scores displayed for predictions
- Aspect-ratio preserving preprocessing for uploaded images
- Helpful error messages and model-loading checks

## 🛠 Tech stack

- Python 3.11
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow (PIL)

## Project structure (recommended)

Below is a suggested project layout and a line-by-line explanation of each file/folder. Adopt the tree that best matches your workflow — this repo currently contains the minimal files needed for inference.

Recommended tree:

```
AI-Powered-Brain-Tumor-Detection-System/
├─ app.py                      # Streamlit web app (entry point)
├─ requirements.txt            # Pin minimal dependencies for deployment
├─ README.md                   # Project documentation (this file)
├─ convert_to_onnx.py          # Optional: convert .h5 to .onnx (run locally)
├─ predict.py                  # Optional: CLI for single-image predictions
├─ scripts/                    # Helper scripts (download model, setup, etc.)
├─ models/                     # Optional: place model.onnx here (ignored or LFS)
│  └─ model.onnx               # ONNX model for cloud-friendly inference
├─ weights/                    # Optional local storage for .h5 (gitignored)
│  └─ brain_tumor_detection_model.h5
├─ tests/                      # Unit tests (preprocess, predict, I/O)
└─ .gitignore
```

Line-by-line explanation:

- `app.py` — The Streamlit application. Handles user interface, file upload, preprocessing, model loading, inference, and displaying results. Keep UI-specific logic here. Use helper modules for complex code if needed.
- `requirements.txt` — List of packages required to run the app in production (Streamlit Cloud or other host). Prefer minimal, compatible pins (avoid installing TensorFlow on Streamlit Cloud). Use `onnxruntime` for cloud-friendly inference.
- `README.md` — This documentation. Describe setup, run instructions, model format, and known issues.
- `convert_to_onnx.py` — Optional helper script to convert a Keras `.h5` model to `model.onnx` using `tf2onnx` or similar. Run this locally where TensorFlow is installed, then commit `model.onnx` (or host externally).
- `predict.py` — Optional CLI for batch or single-image prediction without Streamlit. Useful for CI and smoke-testing.
- `scripts/` — Small utility scripts: `download_model.py`, `prepare_data.sh`, environment setup helpers.
- `models/` — Recommended place to store small ONNX models checked into the repo. For large model files, prefer external hosting or Git LFS.
- `weights/` — Local Keras `.h5` weights. This folder should be added to `.gitignore` to avoid large commits.
- `tests/` — Unit tests for `preprocess_image`, input/output shapes, and CLI functions. Use `pytest` for a lightweight test harness.
- `.gitignore` — Exclude virtual environments, model binaries, and editor folders. Example entries below.

Example `.gitignore` snippet (add to repo root):

```
# Python
__pycache__/
*.py[cod]
.venv/
.env

# Model / weights
brain_tumor_detection_model.h5
*.h5
model.onnx

# Editor
.vscode/
```

Notes on models and deployment:

- For Streamlit Cloud, avoid `tensorflow` in `requirements.txt` if you also need `streamlit` since protobuf version conflicts can occur. Use `onnxruntime` instead and serve `model.onnx`.
- If you must store models in the repo, prefer Git LFS for `.h5` files: `git lfs install && git lfs track "*.h5"` and commit the `.gitattributes` file.
- Alternatively, host the model on a cloud bucket or Hugging Face Hub and download it at app startup (see `scripts/download_model.py`).

## Quickstart (Windows PowerShell)

1. Clone or open this project folder and select the Python interpreter you want to use in VS Code (Command Palette → "Python: Select Interpreter").

2. Create and activate a virtual environment (recommended):

```powershell
& "C:\Program Files\Python311\python.exe" -m venv .venv
\.venv\Scripts\Activate.ps1
```

3. Upgrade packaging tools and install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements.txt
```

4. Place your trained Keras model in the project root as `brain_tumor_detection_model.h5`.

5. Run the Streamlit app:

```powershell
python -m streamlit run app.py
```

Open the URL printed by Streamlit (typically http://localhost:8501).

## Input / Output

- Input: JPG or PNG MRI images. The app converts images to RGB and resizes/pads them to the model input shape.
- Output: Predicted label (e.g., "No Tumor" / tumor type) and a confidence score. The app also shows raw prediction values for debugging.

## Model expectations

- Model input: (H, W, C) RGB images. If your model expects a different shape or preprocessing, update `app.py` accordingly.
- Model output: either a single sigmoid probability (binary classifier) or a softmax vector (multi-class). The app attempts to interpret both formats.

## Troubleshooting

- Pylance / VS Code import errors: ensure VS Code uses the same Python interpreter where packages are installed (Command Palette → "Python: Select Interpreter").
- TensorFlow/protobuf errors: use a fresh virtual environment and install `tensorflow` first so pip can select compatible dependency versions.
- `pip ResolutionImpossible`: create a new venv or use conda/mamba to avoid conflicting global packages.

## Important: model files & git

Do NOT commit large model binaries into the repository. Instead:

- Add `brain_tumor_detection_model.h5` to `.gitignore`.
- Use Git LFS if you must store large binaries in GitHub: `git lfs install && git lfs track "*.h5"`.
- Prefer cloud storage (S3, Google Drive, Hugging Face Hub) for model artifacts and provide a small download script.

## Contributing

Small, focused PRs are welcome. Ideas:

- Add a CLI script for batch predictions (`predict.py`).
- Add unit tests for `preprocess_image` and model I/O.
- Improve UI/UX (progress bars, sample images, help text).

To contribute: fork → create a branch → open a pull request.

## License

Add a license file before publishing (MIT / Apache-2.0 recommended).

---

If you'd like, I can also create a small `predict.py` script and a `.gitignore` for you, or push these changes to GitHub. Tell me which tasks to do next.
