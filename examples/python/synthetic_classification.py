#!/usr/bin/env python3
"""Tier 1: synthetic classification train/infer (offline, fast)."""

from __future__ import annotations

import sys

import numpy as np

import hypercube_cnn as hc


def main() -> int:
    rng = np.random.default_rng(0)
    net = hc.HCNNConfig(
        dim=6,
        num_outputs=3,
        num_threads=1,
        layers=[
            hc.LayerSpec.conv(8),
            hc.LayerSpec.pool("max"),
            hc.LayerSpec.conv(8),
        ],
        weight_seed=1,
    ).build()

    B = 48
    X = rng.standard_normal((B, net.N), dtype=np.float32)
    y = rng.integers(0, 3, size=B, dtype=np.int32)

    def accuracy() -> float:
        pred = np.array([net.predict_class(X[i]) for i in range(B)])
        return float(np.mean(pred == y))

    acc0 = accuracy()
    params = hc.TrainParams(learning_rate=1e-2)
    for epoch in range(8):
        params.shuffle_seed = epoch + 1
        net.train_epoch(X, y, batch_size=16, params=params)
    acc1 = accuracy()
    print(f"synthetic_classification: acc {acc0:.3f} -> {acc1:.3f}")
    if acc1 <= acc0:
        print("ERROR: expected accuracy to improve", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
