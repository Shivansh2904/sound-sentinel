"""Tests for feature extraction and the ONNX export step in train.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train  # noqa: E402

N_FEATURES = 326


def toy_data(n_classes: int = 3, per_class: int = 10, seed: int = 0):
    """Small, easily separable data with the same width as the real features."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), per_class)
    X = rng.normal(size=(len(y), N_FEATURES)).astype(np.float32)
    X += y[:, None].astype(np.float32)
    return X, y


class SimulatedConverterError(Exception):
    pass


@pytest.fixture
def converter_fails(monkeypatch):
    """Make skl2onnx's converter raise, whatever model it is given.

    The failure tests below use the SVM pipeline, which skl2onnx converts
    cleanly (see test_writes_a_model_that_matches_sklearn). The only reason
    they fail is this patch, so they keep testing the error handling even
    after the real model becomes exportable.
    """
    import skl2onnx

    def convert_sklearn(*args, **kwargs):
        raise SimulatedConverterError("simulated converter failure")

    monkeypatch.setattr(skl2onnx, "convert_sklearn", convert_sklearn)


def run_main(tmp_path, monkeypatch, per_class: int = 15):
    """Run train.main() on toy data with the SVM pipeline standing in for the ensemble."""
    X, y = toy_data(per_class=per_class)
    monkeypatch.setattr(train, "load_dataset", lambda data_dir: (X, y, ["a", "b", "c"]))
    monkeypatch.setattr(train, "build_ensemble", lambda n_classes: train.build_svm())
    models_dir = tmp_path / "models"
    onnx_path = tmp_path / "public" / "model.onnx"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--no-cv",
        "--output-dir", str(models_dir),
        "--onnx-path", str(onnx_path),
    ])
    train.main()
    return models_dir, onnx_path


class TestExtractFeatures:
    def test_returns_326_finite_values(self, tmp_path):
        import soundfile as sf

        sr = 22050
        t = np.arange(2 * sr) / sr
        path = tmp_path / "tone.wav"
        sf.write(str(path), 0.3 * np.sin(2 * np.pi * 440 * t), sr)

        features = train.extract_features(str(path))

        assert features.shape == (N_FEATURES,)
        assert np.all(np.isfinite(features))


class TestExportToOnnx:
    def test_writes_a_model_that_matches_sklearn(self, tmp_path):
        import onnxruntime as ort

        X, y = toy_data()
        model = train.build_svm().fit(X, y)
        out = tmp_path / "model.onnx"

        train.export_to_onnx(model, N_FEATURES, out)

        assert out.exists()
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        labels = session.run(None, {session.get_inputs()[0].name: X})[0]
        np.testing.assert_array_equal(labels, model.predict(X))

    def test_raises_when_conversion_fails(self, tmp_path, converter_fails):
        # The export has to fail loudly and must not leave a model file behind.
        X, y = toy_data()
        model = train.build_svm().fit(X, y)
        out = tmp_path / "model.onnx"

        with pytest.raises(RuntimeError, match="ONNX export failed") as excinfo:
            train.export_to_onnx(model, N_FEATURES, out)

        assert isinstance(excinfo.value.__cause__, SimulatedConverterError)
        assert not out.exists()


class TestMain:
    def test_exits_nonzero_when_onnx_export_fails(self, tmp_path, monkeypatch, capsys, converter_fails):
        with pytest.raises(SystemExit) as excinfo:
            run_main(tmp_path, monkeypatch)

        assert excinfo.value.code == 1
        models_dir = tmp_path / "models"
        onnx_path = tmp_path / "public" / "model.onnx"
        assert (models_dir / "sound_classifier.pkl").exists()
        assert not onnx_path.exists()
        out = capsys.readouterr().out
        assert "simulated converter failure" in out
        assert "no ONNX model was written" in out
        assert "Training complete!" not in out

    def test_writes_onnx_when_export_succeeds(self, tmp_path, monkeypatch, capsys):
        models_dir, onnx_path = run_main(tmp_path, monkeypatch)

        assert (models_dir / "sound_classifier.pkl").exists()
        assert onnx_path.exists()
        assert "Training complete!" in capsys.readouterr().out
