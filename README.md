# SoundSentinel

[![CI](https://img.shields.io/github/actions/workflow/status/Shivansh2904/sound-sentinel/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/Shivansh2904/sound-sentinel/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

SoundSentinel classifies environmental sounds into the 50 classes of the
[ESC-50](https://github.com/karolpiczak/ESC-50) dataset. The model is trained in Python with
librosa features and scikit-learn/XGBoost. The goal is to run it entirely in the browser: capture
microphone audio, compute the same features in a Web Worker written in TypeScript, and classify
them with an ONNX model through ONNX Runtime Web.

## Status: work in progress

**The browser model is not shipped yet.** There is no `public/model.onnx` in this repository, and
`train.py` cannot export one yet (see below). Without it, the web app shows a "Model not loaded"
message and recording stays disabled.

What works today:

- `training/download_esc50.py` downloads ESC-50 and unpacks it into `data/ESC-50-master/`.
- `training/train.py` extracts a 326-value feature vector per clip with librosa, trains an
  SVM + XGBoost soft-voting ensemble, prints a classification report on a hold-out split, and saves
  the fitted model to `training/models/sound_classifier.pkl`. It then attempts the ONNX export,
  reports that it failed, and exits with status 1.
- The web app type-checks, passes ESLint (`npm run lint`) and builds (`npm run build`). The Web
  Worker contains a TypeScript port of the feature extraction (FFT, mel filterbank, DCT for the
  MFCCs, spectral centroid, rolloff and zero-crossing rate) and the ONNX Runtime Web inference code.
- `make test` runs the Python tests (feature extraction, the ESC-50 label map, the ONNX export
  step, how `evaluate.py` labels its score, and the downloader, all without the dataset) and two
  small JavaScript checks.

Known gaps, in the order they need fixing:

1. **ONNX export.** skl2onnx cannot convert the `VotingClassifier` (it does not support
   `flatten_transform=True`) and has no converter registered for `XGBClassifier`. An SVM-only
   pipeline does convert, and the tests check that its ONNX labels match scikit-learn's.
2. **The worker cannot load ONNX Runtime Web's runtime files.** It sets
   `ort.env.wasm.wasmPaths = "/"`, so ONNX Runtime Web requests its `.mjs` and `.wasm` files from
   the site root, and nothing serves them there: `vite build` emits the `.wasm` into `dist/assets/`
   under a hashed name, and there is no `public/` directory.
3. **Browser features do not match the training features yet.** The TypeScript port differs from
   librosa in several places: mel scale and filter normalisation, the number of mel bands the MFCCs
   are computed from, frame centring, the dB reference, and whether centroid and rolloff use the
   magnitude or the power spectrum. The app also sends one-second windows, while the model is
   trained on five-second clips. A test comparing the worker's output with librosa is needed
   before the browser predictions can mean anything.
4. **`predict.py` and `benchmark.py` do not work with the trained model.** They expect a different
   file name, a bare estimator rather than the dictionary `train.py` saves, and an 80-value MFCC
   summary rather than the 326-value vector.
5. **Evaluation.** `train.py` refits the saved model on every clip, and `evaluate.py` then scores
   one ESC-50 fold with it (fold 5 by default), so it measures training accuracy, not test
   accuracy. The hold-out split in `train.py` is random (stratified by class) and ignores ESC-50's
   folds, so clips cut from the same source recording can end up on both sides of it.

No accuracy figure is given here. One will be added once evaluation uses ESC-50's official folds,
together with the script that produces it.

## How it works

### Training (`training/train.py`)

Each clip is loaded at 22,050 Hz and padded or trimmed to 5 seconds. Features use librosa's default framing
(2,048-sample FFT, 512-sample hop) and are summarised over time:

| Features | Values |
|---|---|
| 40 MFCCs: mean, std, min, max | 160 |
| 40-band log-mel spectrogram: mean, std, min, max per band | 160 |
| Spectral centroid, spectral rolloff (85%), zero-crossing rate: mean and std | 6 |
| **Total** | **326** |

The classifier is a soft-voting ensemble of two pipelines, each with a `StandardScaler`: an RBF SVM
(`C=10`, `gamma="scale"`) and XGBoost (300 trees, depth 6).

### Browser (`src/`)

- `App.tsx` captures microphone audio with the WebAudio API into a ring buffer and draws the
  waveform on a canvas.
- Once a second it sends the buffered audio to `worker/inference.worker.ts`, which resamples it to
  22,050 Hz, computes the feature vector and runs `public/model.onnx` with ONNX Runtime Web (WASM).
  The app shows the top three classes.
- Audio is never uploaded. Capture, feature extraction and inference all happen in the page.

## Getting started

Requirements: Python 3.11, Node.js 20 or later, and `make` (optional, the commands behind each
target are in the `Makefile`).

```bash
git clone https://github.com/Shivansh2904/sound-sentinel.git
cd sound-sentinel
make install      # pip install -r training/requirements.txt, then npm ci
```

### Get the data

```bash
make download     # same as: python training/download_esc50.py --dest ./data
```

This downloads the ESC-50 archive from GitHub (a large download), unpacks it into
`data/ESC-50-master/` and checks that the audio and `meta/esc50.csv` are present. `data/` is
git-ignored.

### Train

```bash
cd training
python train.py --data-dir ../data/ESC-50-master --output-dir models --onnx-path ../public/model.onnx
```

`--no-cv` skips the extra cross-validation of the SVM on the training split. As described above, the
script saves `training/models/sound_classifier.pkl` and then exits with status 1 because the ONNX
export fails.

### Tests

```bash
make test         # pytest in training/, then npm test
```

### Web app

```bash
npm run dev       # http://localhost:5173
```

Until `public/model.onnx` exists, the page shows the "Model not loaded" message.

### Notebook

[`training/notebooks/audio_exploration.ipynb`](training/notebooks/audio_exploration.ipynb) plots a
clip's waveform, mel spectrogram and MFCCs. Run `make download` first; it reads clips from
`data/ESC-50-master/audio/`.

## Project layout

```
sound-sentinel/
├── .github/workflows/ci.yml        # pytest, flake8, JS tests, type-check, ESLint, build
├── src/
│   ├── App.tsx                     # UI, microphone capture, worker messaging
│   ├── components/WaveformCanvas.tsx
│   ├── constants/labels.ts         # ESC-50 display names, by target index
│   └── worker/inference.worker.ts  # TypeScript feature extraction + ONNX Runtime Web
├── tests/                          # npm test (node:test)
├── training/
│   ├── download_esc50.py           # fetch and unpack ESC-50 into ./data
│   ├── train.py                    # features, training, ONNX export
│   ├── evaluate.py                 # confusion matrix and per-class plots
│   ├── predict.py                  # single-file CLI (see known gaps)
│   ├── benchmark.py                # latency CLI (see known gaps)
│   ├── notebooks/audio_exploration.ipynb
│   ├── tests/                      # pytest
│   └── requirements.txt
├── Makefile
├── eslint.config.js
├── index.html
├── package.json
└── vite.config.ts
```

## Dataset and licence

This project uses ESC-50, created by Karol J. Piczak. The dataset is not included in this
repository; `download_esc50.py` fetches it from the author's GitHub repository. ESC-50 is
released under the [Creative Commons Attribution-NonCommercial licence](http://creativecommons.org/licenses/by-nc/3.0/)
(the ESC-10 subset is CC BY), and per-clip attributions are in the dataset's `LICENSE` file. Keep
that licence in mind before using the data, or a model trained on it, commercially.

If you use ESC-50, please cite:

> K. J. Piczak. ESC: Dataset for Environmental Sound Classification. *Proceedings of the 23rd
> Annual ACM Conference on Multimedia*, Brisbane, Australia, 2015.
> [doi:10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390)

```bibtex
@inproceedings{piczak2015dataset,
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  doi = {10.1145/2733373.2806390},
  location = {{Brisbane, Australia}},
  publisher = {{ACM Press}},
  pages = {1015--1018}
}
```

## License

The code in this repository is MIT licensed, see [LICENSE](LICENSE). The MIT licence does not
cover ESC-50.
