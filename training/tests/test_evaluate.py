"""Tests for how evaluate.py labels its score."""
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # noqa: E402
import train  # noqa: E402

N_FEATURES = 326


def test_does_not_call_the_score_a_test_score(tmp_path, monkeypatch, capsys):
    # train.py fits the saved model on every clip, so the fold evaluate.py
    # scores was part of the training data. Neither the printed result nor
    # the charts may present that number as test accuracy.
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 10)
    X = (rng.normal(size=(len(y), N_FEATURES)) + y[:, None]).astype(np.float32)
    class_names = ["a", "b", "c"]
    model_path = tmp_path / "model.pkl"
    joblib.dump(
        {"model": train.build_svm().fit(X, y), "class_names": class_names, "n_features": N_FEATURES},
        model_path,
    )
    monkeypatch.setattr(evaluate, "load_test_features", lambda data_dir, test_fold: (X, y, class_names))

    titles = []
    close = evaluate.plt.close

    def record_titles_then_close(fig):
        titles.extend(ax.get_title() for ax in fig.axes)
        close(fig)

    monkeypatch.setattr(evaluate.plt, "close", record_titles_then_close)
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--model-path", str(model_path),
        "--output-dir", str(tmp_path / "outputs"),
        "--test-fold", "3",
    ])

    evaluate.main()

    out = capsys.readouterr().out
    assert "Test Accuracy" not in out
    assert "Accuracy on fold 3" in out
    assert "training accuracy" in out
    chart_titles = [t for t in titles if t]
    assert len(chart_titles) == 2
    assert not any("Test" in t for t in chart_titles)
    assert all("fold 3" in t for t in chart_titles)
