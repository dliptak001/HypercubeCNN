#!/usr/bin/env python3
"""Tier 1: synthetic regression train/infer (offline, fast)."""

from __future__ import annotations

import sys

import numpy as np

import hypercube_cnn as hc


def main() -> int:
    rng = np.random.default_rng(1)
    net = hc.HCNNConfig(
        dim=6,
        num_outputs=2,
        task=hc.TaskType.Regression,
        num_threads=1,
        layers=[hc.LayerSpec.conv(8, activation="tanh")],
        weight_seed=2,
    ).build()

    B = 48
    X = rng.standard_normal((B, net.N), dtype=np.float32)
    Y = np.stack([X.mean(axis=1), X.std(axis=1)], axis=1).astype(np.float32)

    def mse() -> float:
        pred = np.stack([net.predict(X[i]) for i in range(B)], axis=0)
        return float(np.mean((pred - Y) ** 2))

    m0 = mse()
    params = hc.TrainParams(learning_rate=5e-3)
    for epoch in range(10):
        params.shuffle_seed = epoch + 1
        net.train_epoch(X, Y, batch_size=16, params=params)
    m1 = mse()
    print(f"synthetic_regression: mse {m0:.6f} -> {m1:.6f}")
    if m1 >= m0 * 0.95:
        print("ERROR: expected MSE to drop", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
