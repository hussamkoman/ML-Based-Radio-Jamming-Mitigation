#!/usr/bin/env python3
"""
inference_benchmark.py

Measure throughput & latency of the trained QAM-receiver on RTX3070.
Generates:
  - latency_histogram.png
  - throughput_vs_batch.png
  - inference_results.csv
"""

import time
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# CONFIGURATION
MODEL_PATH = "ml/007_trained_on_dataset_07.keras"
OUTPUT_DIR = Path("plots/benchmark_results/")
BATCH_SIZES = [1, 4, 16, 32]
BLOCK_LEN = 16384  # samples per block
N_RUNS = 1000
WARMUP = 50

OUTPUT_DIR.mkdir(exist_ok=True)


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)

    @tf.function
    def infer(x):
        return model(x, training=False)

    return infer


def make_dummy(batch_size: int):
    x = np.random.randn(batch_size, BLOCK_LEN, 2).astype(np.float32)
    return tf.constant(x)


def benchmark(infer, batch_size: int):
    inp = make_dummy(batch_size)
    # warmup
    for _ in range(WARMUP):
        _ = infer(inp)
    # timed loop
    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = infer(inp)
        latencies.append((time.perf_counter() - t0) * 1e3)  # ms
    lat_arr = np.array(latencies)
    p50, p99 = np.percentile(lat_arr, [50, 99])
    mean_lat = lat_arr.mean()
    throughput = batch_size / (mean_lat / 1e3)
    return mean_lat, p50, p99, throughput, lat_arr


def main():
    infer = load_model()
    rows = []
    for bs in BATCH_SIZES:
        mean_lat, p50, p99, tp, lat_arr = benchmark(infer, bs)
        rows.append([bs, mean_lat, p50, p99, tp])
        # save histogram
        plt.figure()
        plt.hist(lat_arr, bins=50)
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        plt.title(f"Latency Distribution (batch={bs})")
        plt.savefig(OUTPUT_DIR / f"latency_hist_bs{bs}.png", dpi=150)
        plt.close()
    # throughput plot
    bs_list, _, _, _, tp_list = zip(*rows)
    plt.figure()
    plt.plot(bs_list, tp_list, marker='o')
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (blocks/s)")
    plt.title("Throughput vs Batch Size")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "throughput_vs_batch.png", dpi=150)
    plt.close()
    # dump CSV
    with open(OUTPUT_DIR / "inference_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "mean_lat_ms", "p50_ms", "p99_ms", "throughput_blocks_per_s"])
        w.writerows(rows)
    print("Done. Results in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
