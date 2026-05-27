# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Weekly Dependabot updates for pip (`/training`), npm (`/`), and GitHub Actions
- `CONTRIBUTING.md` with training pipeline + browser frontend setup

## [1.3.0] — 2026-05-27

### Added
- `training/download_esc50.py` — standalone downloader for the ESC-50 dataset with progress bar, idempotent extraction, and a verifier (2000 wavs + metadata)
- `training/notebooks/audio_exploration.ipynb` — 5-section walkthrough of waveform, mel-spectrogram, MFCC, and the 80-dim feature vector used by the classifier

## [1.2.0] — 2026-05-27

### Added
- `training/tests/test_predict.py` — 8 pytest tests covering `LABEL_MAP` shape/content and `extract_features` (80-dim output, finite values, short-audio padding)
- "Run unit tests" step in CI

### Fixed
- CI: cast `self` to `DedicatedWorkerGlobalScope` for the `postMessage` transfer-list call in the inference worker
- CI: added `WebWorker` to the tsconfig `lib` array so `DedicatedWorkerGlobalScope` resolves
- CI: added `package-lock.json` for npm cache

## [1.1.0] — 2026-05-27

### Added
- `training/predict.py` — single-file inference script with `--json` output and ASCII top-5 probability bar chart
- 5-fold cross-validation block in `training/train.py` (in addition to the held-out test evaluation)

## [1.0.0] — 2026-05-17

### Added
- Initial release
- Python ML training pipeline: MFCC + SVM/XGBoost ensemble on ESC-50 (target 95.8% accuracy), ONNX export
- React + TypeScript browser frontend with real-time microphone inference via ONNX Runtime Web
- Hand-rolled FFT + Mel filterbank + DCT-II in a Web Worker (matches Python feature pipeline byte-for-byte)
- Waveform canvas visualization, label overlay
- Docker, GitHub Actions CI
