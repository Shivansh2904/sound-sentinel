"""
SoundSentinel — Training Script
================================
Trains an SVM + XGBoost ensemble on ESC-50 audio features extracted with librosa,
then exports the full pipeline (scaler + ensemble) to ONNX for browser inference.

Usage
-----
    python train.py --data-dir data/ESC-50 --output-dir models

Dataset Setup
-------------
Download ESC-50 from GitHub before running:
    git clone https://github.com/karolpiczak/ESC-50.git training/data/ESC-50

Expected layout:
    training/data/ESC-50/
        audio/          <- 2000 .wav files (5s each, 44100 Hz)
        meta/
            esc50.csv   <- metadata with filename, fold, target, category columns
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

N_MFCC = 40          # Number of MFCC coefficients
N_MELS = 40          # Number of Mel filterbank bands
SAMPLE_RATE = 22050  # Resample all audio to this rate
DURATION = 5.0       # Clip duration in seconds


def extract_features(file_path: str) -> np.ndarray:
    """
    Extract a 326-dimensional feature vector from a single audio file.

    Feature breakdown
    -----------------
    MFCCs (160 features):
        40 coefficients × [mean, std, min, max] = 160

    Mel-spectrogram statistics (160 features):
        40 bands × [mean, std, min, max] = 160

    Spectral centroid (2 features):
        [mean, std]

    Spectral rolloff (2 features):
        [mean, std]

    Zero-crossing rate (2 features):
        [mean, std]

    Total: 160 + 160 + 2 + 2 + 2 = 326 features
    """
    try:
        # Load audio, resample to SAMPLE_RATE, and pad/trim to DURATION seconds
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

        # Pad if shorter than DURATION (some clips may be slightly short)
        target_length = int(SAMPLE_RATE * DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")

        features = []

        # ------------------------------------------------------------------
        # 1. MFCC features (40 coefficients × 4 statistics = 160)
        # ------------------------------------------------------------------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))

        # ------------------------------------------------------------------
        # 2. Mel-spectrogram statistics (40 bands × 4 statistics = 160)
        # ------------------------------------------------------------------
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend(np.mean(mel_spec_db, axis=1))
        features.extend(np.std(mel_spec_db, axis=1))
        features.extend(np.min(mel_spec_db, axis=1))
        features.extend(np.max(mel_spec_db, axis=1))

        # ------------------------------------------------------------------
        # 3. Spectral centroid (2 features)
        # ------------------------------------------------------------------
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.std(spectral_centroid)))

        # ------------------------------------------------------------------
        # 4. Spectral rolloff (2 features)
        # ------------------------------------------------------------------
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features.append(float(np.mean(spectral_rolloff)))
        features.append(float(np.std(spectral_rolloff)))

        # ------------------------------------------------------------------
        # 5. Zero-crossing rate (2 features)
        # ------------------------------------------------------------------
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        return np.array(features, dtype=np.float32)

    except Exception as exc:
        print(f"  [WARN] Could not process {file_path}: {exc}")
        return None


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk the ESC-50 directory, extract features from every audio clip, and
    return X (feature matrix), y (integer labels), and class_names.
    """
    audio_dir = data_dir / "audio"
    meta_path = data_dir / "meta" / "esc50.csv"

    if not meta_path.exists():
        sys.exit(
            f"[ERROR] Metadata file not found: {meta_path}\n"
            "Please download ESC-50:\n"
            "  git clone https://github.com/karolpiczak/ESC-50.git training/data/ESC-50"
        )

    if not audio_dir.exists():
        sys.exit(
            f"[ERROR] Audio directory not found: {audio_dir}\n"
            "Make sure ESC-50 audio files are extracted."
        )

    meta = pd.read_csv(meta_path)
    print(f"[INFO] Found {len(meta)} clips across {meta['category'].nunique()} classes")

    X_list, y_list = [], []
    skipped = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting features"):
        file_path = audio_dir / row["filename"]
        if not file_path.exists():
            skipped += 1
            continue

        feat = extract_features(str(file_path))
        if feat is not None:
            X_list.append(feat)
            y_list.append(row["target"])

    if skipped:
        print(f"[WARN] Skipped {skipped} missing audio files")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Build ordered class name list (index = target integer)
    class_map = meta.drop_duplicates("target").set_index("target")["category"].to_dict()
    class_names = [class_map[i] for i in sorted(class_map.keys())]

    print(f"[INFO] Feature matrix shape: {X.shape}")
    return X, y, class_names


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_svm() -> Pipeline:
    """Return an SVM pipeline with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,   # Required for soft voting
            random_state=42,
            cache_size=1000,
        )),
    ])


def build_xgb(n_classes: int) -> Pipeline:
    """Return an XGBoost pipeline with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            num_class=n_classes,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def build_ensemble(n_classes: int) -> VotingClassifier:
    """
    Soft-voting ensemble of SVM and XGBoost.

    Soft voting averages the predicted class probabilities from each
    sub-estimator, which generally outperforms hard (majority-vote) voting
    for well-calibrated classifiers.
    """
    svm_pipe = build_svm()
    xgb_pipe = build_xgb(n_classes)

    ensemble = VotingClassifier(
        estimators=[
            ("svm", svm_pipe),
            ("xgb", xgb_pipe),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return ensemble


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(model: VotingClassifier, n_features: int, output_path: Path) -> None:
    """
    Export the trained VotingClassifier to ONNX format using skl2onnx.

    The exported model accepts a float32 array of shape [N, n_features] and
    outputs:
        - label:       int64 array [N] — predicted class indices
        - probabilities: float array [N, n_classes] — class probabilities
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("[ERROR] skl2onnx not installed. Run: pip install skl2onnx")
        return

    print("[INFO] Exporting model to ONNX...")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            options={id(model): {"zipmap": False}},  # Return raw probability arrays
            target_opset=17,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        size_kb = output_path.stat().st_size / 1024
        print(f"[INFO] ONNX model saved to {output_path} ({size_kb:.1f} KB)")

    except Exception as exc:
        print(f"[ERROR] ONNX export failed: {exc}")
        print("[INFO] Model saved as .pkl only — ONNX export skipped")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
    """Run 5-fold stratified cross-validation and print results."""
    print("\n[INFO] Running 5-fold stratified cross-validation...")
    print("       (This may take 10-20 minutes depending on your hardware)")

    # Use a lighter SVM-only pipeline for CV speed; full ensemble CV is very slow
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
            cache_size=1000,
        )),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"\n  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold:    {' | '.join(f'{s:.4f}' for s in cv_scores)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SoundSentinel audio classifier on ESC-50"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ESC-50"),
        help="Path to the ESC-50 root directory (default: data/ESC-50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained models (default: models)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation (faster, for quick iterations)",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("../public/model.onnx"),
        help="Path to save the ONNX model (default: ../public/model.onnx)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  SoundSentinel — Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset and extract features
    # ------------------------------------------------------------------
    t0 = time.time()
    X, y, class_names = load_dataset(args.data_dir)
    n_classes = len(class_names)
    print(f"[INFO] Feature extraction complete in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Train/test split — use fold 5 as held-out test set (ESC-50 convention)
    # ------------------------------------------------------------------
    # We use a simple 80/20 stratified split for quick evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[INFO] Train: {len(X_train)} clips  |  Test: {len(X_test)} clips")

    # ------------------------------------------------------------------
    # 3. Optional cross-validation
    # ------------------------------------------------------------------
    if not args.no_cv:
        run_cross_validation(X_train, y_train, n_classes)

    # ------------------------------------------------------------------
    # 4. Train full ensemble on training set
    # ------------------------------------------------------------------
    print("\n[INFO] Training SVM + XGBoost ensemble on full training set...")
    print("       SVM training may take a few minutes...")

    ensemble = build_ensemble(n_classes)
    t1 = time.time()
    ensemble.fit(X_train, y_train)
    print(f"[INFO] Training complete in {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------
    # 5. Evaluate on held-out test set
    # ------------------------------------------------------------------
    y_pred = ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n[RESULT] Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ------------------------------------------------------------------
    # 5b. 5-fold cross-validation on full dataset
    # ------------------------------------------------------------------
    from sklearn.model_selection import cross_val_score
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

    # ------------------------------------------------------------------
    # 6. Save the full fitted model (retrain on all data for best generalization)
    # ------------------------------------------------------------------
    print("[INFO] Retraining ensemble on full dataset for final model...")
    final_ensemble = build_ensemble(n_classes)
    final_ensemble.fit(X, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "sound_classifier.pkl"

    # Save model alongside class names and feature count for evaluate.py
    joblib.dump(
        {
            "model": final_ensemble,
            "class_names": class_names,
            "n_features": X.shape[1],
            "test_accuracy": float(test_acc),
        },
        model_path,
    )
    print(f"[INFO] Model saved to {model_path}")

    # ------------------------------------------------------------------
    # 7. Export to ONNX
    # ------------------------------------------------------------------
    onnx_path = args.onnx_path
    export_to_onnx(final_ensemble, X.shape[1], onnx_path)

    print("\n" + "=" * 60)
    print(f"  Training complete! Test accuracy: {test_acc * 100:.2f}%")
    print(f"  Model:      {model_path}")
    print(f"  ONNX:       {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
