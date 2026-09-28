"""Tests for download_esc50.py that do not touch the network."""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import download_esc50  # noqa: E402


def test_final_hint_points_train_py_at_the_extracted_data(tmp_path, monkeypatch, capsys):
    # A tiny archive laid out like the real ESC-50 zip. Because the zip is
    # already there, main() skips the download and only extracts it.
    dest = tmp_path / "data"
    dest.mkdir()
    with zipfile.ZipFile(dest / "ESC-50-master.zip", "w") as zf:
        zf.writestr("ESC-50-master/meta/esc50.csv", "filename,fold,target,category,esc10,src_file,take\n")
        zf.writestr("ESC-50-master/audio/1-100032-A-0.wav", b"")

    def no_network(*args, **kwargs):
        raise AssertionError("the test must not download anything")

    monkeypatch.setattr(urllib.request, "urlretrieve", no_network)
    monkeypatch.setattr(sys, "argv", ["download_esc50.py", "--dest", str(dest)])

    download_esc50.main()

    hint = capsys.readouterr().out.strip().splitlines()[-1]
    assert "train.py --data-dir" in hint
    data_dir = Path(hint.split("--data-dir", 1)[1].strip())
    # train.py's load_dataset() looks for exactly these two paths.
    assert (data_dir / "meta" / "esc50.csv").is_file()
    assert (data_dir / "audio").is_dir()
