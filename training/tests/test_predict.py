"""Tests for the predict.py inference script."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Make predict importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import extract_features, LABEL_MAP


class TestLabelMap:
    def test_has_50_labels(self):
        assert len(LABEL_MAP) == 50

    def test_label_keys_are_consecutive_ints(self):
        keys = sorted(LABEL_MAP.keys())
        assert keys == list(range(50))

    def test_all_values_are_strings(self):
        for v in LABEL_MAP.values():
            assert isinstance(v, str)
            assert len(v) > 0

    def test_known_labels(self):
        # Spot-check a few mappings from the ESC-50 dataset
        assert LABEL_MAP[0] == "dog"
        assert LABEL_MAP[5] == "cat"
        assert LABEL_MAP[40] == "chainsaw"
        assert LABEL_MAP[49] == "street_music"


class TestExtractFeatures:
    @pytest.fixture
    def sample_wav(self, tmp_path):
        """Create a 2-second mono 22050Hz sine wave for testing."""
        import soundfile as sf
        sr = 22050
        duration = 2.0
        freq = 440.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        path = tmp_path / "test.wav"
        sf.write(str(path), audio, sr)
        return str(path)

    def test_returns_numpy_array(self, sample_wav):
        features = extract_features(sample_wav)
        assert isinstance(features, np.ndarray)

    def test_returns_80_dim_vector(self, sample_wav):
        # 40 mean + 40 std = 80
        features = extract_features(sample_wav)
        assert features.shape == (80,)

    def test_features_are_finite(self, sample_wav):
        features = extract_features(sample_wav)
        assert np.all(np.isfinite(features))

    def test_pads_short_audio(self, tmp_path):
        """A 0.1s clip should still produce an 80-dim vector (padded internally)."""
        import soundfile as sf
        sr = 22050
        audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr * 0.1)))
        path = tmp_path / "short.wav"
        sf.write(str(path), audio, sr)
        features = extract_features(str(path))
        assert features.shape == (80,)
