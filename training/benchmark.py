#!/usr/bin/env python3
"""Benchmark inference latency and throughput of the trained classifier.

Usage:
    python training/benchmark.py --audio path/to/test.wav --iterations 200
    python training/benchmark.py --audio path/to/test.wav --model models/ensemble_model.joblib
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Make predict importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import extract_features  # type: ignore


def benchmark(model_path: str, audio_path: str, iterations: int = 100) -> dict:
    import joblib

    print(f"Loading model: {model_path}")
    t0 = time.perf_counter()
    model = joblib.load(model_path)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  load time: {load_ms:.1f} ms")

    print(f"\nExtracting features from: {audio_path}")
    t0 = time.perf_counter()
    features = extract_features(audio_path).reshape(1, -1)
    feature_ms = (time.perf_counter() - t0) * 1000
    print(f"  feature extraction: {feature_ms:.1f} ms")

    print(f"\nWarming up (10 iterations)...")
    for _ in range(10):
        model.predict(features)

    print(f"\nBenchmarking {iterations} iterations...")
    timings_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(features)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "iterations": iterations,
        "load_time_ms": load_ms,
        "feature_extraction_ms": feature_ms,
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "p99_ms": float(np.percentile(timings_ms, 99)),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "stddev_ms": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "throughput_per_sec": 1000.0 / statistics.mean(timings_ms),
    }


def print_results(results: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"  Inference Benchmark — {results['iterations']} iterations")
    print(f"{'=' * 50}")
    print(f"  Mean:        {results['mean_ms']:.3f} ms")
    print(f"  Median:      {results['median_ms']:.3f} ms")
    print(f"  Stddev:      {results['stddev_ms']:.3f} ms")
    print(f"  Min:         {results['min_ms']:.3f} ms")
    print(f"  Max:         {results['max_ms']:.3f} ms")
    print(f"  P95:         {results['p95_ms']:.3f} ms")
    print(f"  P99:         {results['p99_ms']:.3f} ms")
    print()
    print(f"  Throughput:  {results['throughput_per_sec']:.1f} inferences/sec")
    print(f"  Feat. extr:  {results['feature_extraction_ms']:.1f} ms (one-shot)")
    print(f"  Model load:  {results['load_time_ms']:.1f} ms (one-shot)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sound-sentinel inference")
    parser.add_argument("--audio", required=True, help="Path to a sample audio file")
    parser.add_argument("--model", default="models/ensemble_model.joblib", help="Path to trained model")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    results = benchmark(args.model, args.audio, args.iterations)
    print_results(results)


if __name__ == "__main__":
    main()
