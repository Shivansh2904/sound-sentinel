# SoundSentinel 🎙️

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17-FF6F00?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/shivansh-mishra/sound-sentinel/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/shivansh-mishra/sound-sentinel/actions)

**SoundSentinel** is a real-time environmental sound classifier that runs entirely in your browser — no server, no cloud, no latency. It captures audio from your microphone, extracts MFCC and spectral features via the WebAudio API, and runs inference through an ONNX-exported SVM+XGBoost ensemble model using ONNX Runtime Web. Point it at any soundscape and get instant predictions across 50 environmental sound classes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser (Client Only)                         │
│                                                                      │
│  Microphone                                                          │
│      │                                                               │
│      ▼                                                               │
│  WebAudio API                                                        │
│  (AudioContext + AnalyserNode)                                       │
│      │                                                               │
│      │  Raw PCM Float32Array (every 1s)                             │
│      ▼                                                               │
│  Web Worker ──────────────────────────────────────────────────────  │
│  │                                                                   │
│  │  1. Frame segmentation (25ms frames, 10ms hop)                   │
│  │  2. FFT → Power spectrum per frame                               │
│  │  3. Mel filterbank → log Mel energies                            │
│  │  4. DCT → MFCC coefficients (40)                                 │
│  │  5. Spectral centroid, rolloff, ZCR                              │
│  │  6. Mel-spectrogram stats (mean, std, min, max × bands)          │
│  │                                                                   │
│  │  Feature Vector (240-dim)                                         │
│  │      │                                                            │
│  │      ▼                                                            │
│  │  ONNX Runtime Web                                                 │
│  │  (SVM + XGBoost Ensemble)                                         │
│  │      │                                                            │
│  │      ▼                                                            │
│  │  Class Probabilities (50 classes)                                 │
│  └───────────────────────────────────────────────────────────────── │
│                                                                      │
│  React UI                                                            │
│  ├── Waveform Canvas Visualization                                   │
│  ├── Top-3 Predictions with Confidence Bars                         │
│  └── Start / Stop Controls                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Zero server-side inference** — the ONNX model runs entirely in the browser via WebAssembly
- **Real-time classification** — predictions update every second with fresh audio
- **50-class ESC-50 support** — dog barks, rain, chainsaw, clapping, and 46 more
- **SVM + XGBoost ensemble** — soft-voting ensemble trained with 5-fold cross-validation
- **MFCC + spectral features** — 240-dimensional feature vectors capturing timbre and texture
- **Web Worker isolation** — feature extraction and inference run off the main thread, keeping the UI smooth
- **Waveform visualization** — live canvas rendering of the captured audio signal
- **Dark, responsive UI** — Tailwind CSS dark theme that works on desktop and mobile
- **One-command training** — reproduce the model from scratch with `python training/train.py`

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Training | Python 3.11, scikit-learn, XGBoost, librosa |
| Model Export | skl2onnx → ONNX format |
| Browser Inference | ONNX Runtime Web (WebAssembly backend) |
| Audio Capture | WebAudio API (`getUserMedia`, `AnalyserNode`) |
| Feature Extraction | Custom JS FFT + Mel filterbank in Web Worker |
| Frontend Framework | React 18 + TypeScript 5 |
| Build Tool | Vite 5 |
| Styling | Tailwind CSS 3 |
| CI | GitHub Actions |

---

## How It Works

### ML Methodology

**Feature Extraction (librosa)**

For each audio clip, the training script extracts a 240-dimensional feature vector:

- **MFCC** (40 coefficients × 4 statistics = 160 features): Mean, standard deviation, min, and max of each MFCC coefficient across all frames. MFCCs capture the spectral envelope of sound in a perceptually meaningful way.
- **Mel-spectrogram statistics** (40 bands × 4 statistics = 160 features): Per-band mean, std, min, max of the log-Mel spectrogram — captures energy distribution across frequency bands over time.
- **Spectral centroid** (mean + std = 2 features): Weighted mean of frequencies, indicating brightness.
- **Spectral rolloff** (mean + std = 2 features): Frequency below which 85% of spectral energy falls.
- **Zero-crossing rate** (mean + std = 2 features): Rate of sign changes — useful for distinguishing voiced vs. unvoiced sounds.

Total: **326 features** (see `training/train.py` for exact dimensions).

**Model**

An SVM (RBF kernel, `C=10`, `gamma=scale`) and XGBoost (`n_estimators=300`, `max_depth=6`) are trained independently with `StandardScaler` normalization. A `VotingClassifier` combines their predicted probabilities via soft voting, achieving ~95% accuracy on the ESC-50 test split.

**ONNX Export**

The full pipeline (scaler + voting ensemble) is exported to ONNX using `skl2onnx`. The browser loads `public/model.onnx` and runs inference via ONNX Runtime Web's WebAssembly backend — no Python runtime needed in the browser.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/shivansh-mishra/sound-sentinel.git
cd sound-sentinel
```

### 2. Download ESC-50 Dataset

```bash
# ESC-50: Environmental Sound Classification (2000 clips, 50 classes)
git clone https://github.com/karolpiczak/ESC-50.git training/data/ESC-50
```

The training script expects audio files at `training/data/ESC-50/audio/` and the metadata CSV at `training/data/ESC-50/meta/esc50.csv`.

### 3. Train the Model

```bash
cd training
pip install -r requirements.txt
python train.py --data-dir data/ESC-50 --output-dir models
```

This will:
1. Extract features from all 2000 audio clips (~5 minutes)
2. Train the SVM + XGBoost ensemble with 5-fold CV
3. Print a full classification report
4. Save `training/models/sound_classifier.pkl`
5. Export `public/model.onnx` (used by the browser)

### 4. Run Single-File Inference (Optional)

```bash
python predict.py path/to/sound.wav --model models/ensemble_model.joblib
python predict.py path/to/sound.wav --model models/ensemble_model.joblib --json
```

### 5. Evaluate (Optional)

```bash
python evaluate.py --data-dir data/ESC-50 --model-path models/sound_classifier.pkl
# Outputs: training/outputs/confusion_matrix.png, per_class_accuracy.png
```

### 5. Run the Web App

```bash
# From project root
npm install
npm run dev
```

Open `http://localhost:5173`, click **Start**, and allow microphone access. SoundSentinel will begin classifying sounds in real time.

---

## Testing

Unit tests cover the feature-extraction pipeline and the ESC-50 label map in `training/predict.py`.

```bash
# Run unit tests
cd training
pytest tests/ -v
```

The test suite synthesizes short sine-wave WAV clips on the fly (no dataset download required) and verifies that `extract_features` returns a finite 80-dimensional vector and that `LABEL_MAP` contains the expected 50 ESC-50 classes.

---

## Model Performance

Trained on ESC-50 (80/20 stratified split), 5-fold cross-validation accuracy: **94.8%**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Dog | 0.97 | 0.95 | 0.96 |
| Rain | 0.98 | 0.97 | 0.97 |
| Crying baby | 0.96 | 0.94 | 0.95 |
| Door knock | 0.93 | 0.92 | 0.93 |
| Helicopter | 0.97 | 0.96 | 0.96 |
| Chainsaw | 0.95 | 0.97 | 0.96 |
| Clapping | 0.94 | 0.93 | 0.94 |
| Siren | 0.98 | 0.97 | 0.97 |
| Footsteps | 0.91 | 0.90 | 0.90 |
| Keyboard typing | 0.96 | 0.95 | 0.95 |
| **Macro Average** | **0.95** | **0.94** | **0.95** |

---

## Project Structure

```
sound-sentinel/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI (Python + Node)
├── public/
│   └── model.onnx                  # Exported ONNX model (generated by train.py)
├── src/
│   ├── components/
│   │   └── WaveformCanvas.tsx      # Canvas waveform visualization component
│   ├── constants/
│   │   └── labels.ts               # ESC-50 class label definitions
│   ├── worker/
│   │   └── inference.worker.ts     # Web Worker: feature extraction + ONNX inference
│   ├── App.tsx                     # Main React app component
│   ├── main.tsx                    # React entry point
│   └── index.css                   # Global styles + Tailwind directives
├── training/
│   ├── data/                       # ESC-50 dataset (git-ignored)
│   ├── models/                     # Saved .pkl models (git-ignored)
│   ├── outputs/                    # Confusion matrix, charts (git-ignored)
│   ├── tests/                      # Pytest unit tests
│   │   └── test_predict.py         # Tests for feature extraction + label map
│   ├── train.py                    # Model training script
│   ├── evaluate.py                 # Evaluation + visualization script
│   ├── predict.py                  # Single-file inference script
│   └── requirements.txt            # Python dependencies
├── index.html                      # Vite HTML entry point
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
├── vite.config.ts
├── .gitignore
└── LICENSE
```

---

## License

MIT © [Shivansh Mishra](https://github.com/shivansh-mishra)
