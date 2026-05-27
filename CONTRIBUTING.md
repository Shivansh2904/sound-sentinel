# Contributing to SoundSentinel

Thanks for considering a contribution! This project has two parts: Python training scripts and a React/ONNX Runtime Web frontend that does in-browser inference.

## Training pipeline

```bash
cd training
pip install -r requirements.txt

# Download the ESC-50 dataset (~600 MB, one-time)
python download_esc50.py --dest ../data

# Train the ensemble (SVM + XGBoost on MFCCs)
python train.py

# Evaluate on held-out set
python evaluate.py

# Run inference on a single audio file
python predict.py path/to/sound.wav --model models/ensemble_model.joblib
```

Tests:

```bash
cd training
pytest tests/ -v
```

## Frontend (in-browser inference)

```bash
npm install
npm run dev
# UI: http://localhost:5173
```

The frontend loads the ONNX model directly in a Web Worker (`src/worker/inference.worker.ts`), records mic audio via WebAudio, extracts MFCC features with a hand-rolled DCT, and runs inference with ONNX Runtime Web. No server, no API key.

## Adding a new feature extractor

1. Implement it in `training/train.py` (sklearn-compatible transformer or function)
2. Update `training/predict.py` to match
3. Update `src/worker/inference.worker.ts` to match (must produce identical features client-side)
4. Re-export the ONNX model with the new pipeline
5. Add a test in `training/tests/test_predict.py`

The trickiest part is keeping the Python feature pipeline byte-for-byte equivalent to the TS one — always re-test end-to-end after changes.

## Style

- Python: PEP 8, type hints
- TypeScript: strict mode
- ML code: prefer explicit shapes and dtypes everywhere

## Submitting a PR

1. Fork, branch, commit
2. `pytest training/tests/ -v` passes
3. `npm run build` succeeds
4. Update README if you add a script

## License

By contributing, you agree your contributions are licensed under MIT.
