#!/usr/bin/env python3
"""Download the ESC-50 dataset (audio + metadata) for training.

ESC-50: Dataset for Environmental Sound Classification
2000 5-second clips across 50 classes (40 examples each), released under CC BY-NC.
https://github.com/karolpiczak/ESC-50
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ESC-50-master.zip"

    if zip_path.exists():
        print(f"Zip already exists at {zip_path}, skipping download.")
    else:
        print(f"Downloading ESC-50 (~600 MB) to {zip_path}...")

        def progress(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            pct = min(100.0, 100.0 * downloaded / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r  {pct:5.1f}%  {downloaded / 1_048_576:.1f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(ESC50_URL, zip_path, reporthook=progress)
        print()  # newline after progress

    return zip_path


def extract(zip_path: Path, dest: Path) -> Path:
    audio_dir = dest / "ESC-50-master" / "audio"
    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"Audio directory already exists at {audio_dir}, skipping extraction.")
        return dest / "ESC-50-master"

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    return dest / "ESC-50-master"


def verify(root: Path) -> None:
    audio_dir = root / "audio"
    meta_file = root / "meta" / "esc50.csv"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Expected {audio_dir} not found")
    if not meta_file.exists():
        raise FileNotFoundError(f"Expected {meta_file} not found")

    wav_count = len(list(audio_dir.glob("*.wav")))
    if wav_count != 2000:
        print(f"WARNING: expected 2000 wav files, found {wav_count}")
    else:
        print(f"OK -- found {wav_count} audio clips and metadata.")

    print(f"Audio dir: {audio_dir}")
    print(f"Metadata:  {meta_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESC-50 dataset for training")
    parser.add_argument(
        "--dest",
        default="./data",
        help="Destination directory (default: ./data)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    zip_path = download(dest)
    root = extract(zip_path, dest)
    verify(root)

    if not args.keep_zip and zip_path.exists():
        print(f"Removing zip file {zip_path}")
        zip_path.unlink()

    print("Done. You can now run: python training/train.py")


if __name__ == "__main__":
    main()
