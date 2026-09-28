#!/usr/bin/env python3
"""Run inference on a single audio file using the trained model."""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

# ESC-50 target index -> category, as listed in the dataset's meta/esc50.csv.
LABEL_MAP = {
    0: "dog", 1: "rooster", 2: "pig", 3: "cow", 4: "frog",
    5: "cat", 6: "hen", 7: "insects", 8: "sheep", 9: "crow",
    10: "rain", 11: "sea_waves", 12: "crackling_fire", 13: "crickets", 14: "chirping_birds",
    15: "water_drops", 16: "wind", 17: "pouring_water", 18: "toilet_flush", 19: "thunderstorm",
    20: "crying_baby", 21: "sneezing", 22: "clapping", 23: "breathing", 24: "coughing",
    25: "footsteps", 26: "laughing", 27: "brushing_teeth", 28: "snoring", 29: "drinking_sipping",
    30: "door_wood_knock", 31: "mouse_click", 32: "keyboard_typing", 33: "door_wood_creaks", 34: "can_opening",
    35: "washing_machine", 36: "vacuum_cleaner", 37: "clock_alarm", 38: "clock_tick", 39: "glass_breaking",
    40: "helicopter", 41: "chainsaw", 42: "siren", 43: "car_horn", 44: "engine",
    45: "train", 46: "church_bells", 47: "airplane", 48: "fireworks", 49: "hand_saw",
}


def extract_features(audio_path: str, sr: int = 22050, n_mfcc: int = 40) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, duration=5.0)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])


def predict(model_path: str, audio_path: str) -> dict:
    import joblib
    model = joblib.load(model_path)
    features = extract_features(audio_path).reshape(1, -1)
    pred = int(model.predict(features)[0])

    # Get probabilities if available
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        top5_idx = np.argsort(probs)[::-1][:5]
        proba = {LABEL_MAP.get(i, str(i)): round(float(probs[i]), 4) for i in top5_idx}

    return {
        "predicted_class": LABEL_MAP.get(pred, str(pred)),
        "class_id": pred,
        "top5_probabilities": proba,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sound class for an audio file")
    parser.add_argument("audio", help="Path to .wav or .ogg audio file")
    parser.add_argument("--model", default="models/ensemble_model.joblib", help="Path to trained model")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    result = predict(args.model, args.audio)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Predicted: {result['predicted_class']} (class {result['class_id']})")
        if result["top5_probabilities"]:
            print("\nTop 5:")
            for label, prob in result["top5_probabilities"].items():
                bar = "█" * int(prob * 30)
                print(f"  {label:<20} {bar} {prob:.2%}")


if __name__ == "__main__":
    main()
