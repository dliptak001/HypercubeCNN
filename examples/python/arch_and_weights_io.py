#!/usr/bin/env python3
"""Tier 1: arch JSON + HCNW round-trip (bit-identical logits)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import hypercube_cnn as hc


def main() -> int:
    net = hc.HCNNConfig(
        dim=6,
        num_outputs=3,
        num_threads=1,
        layers=[
            hc.LayerSpec.conv(8, bn=True),
            hc.LayerSpec.pool("max"),
            hc.LayerSpec.conv(8),
        ],
        weight_seed=7,
    ).build()

    x = np.random.default_rng(0).standard_normal(net.N).astype(np.float32)
    logits0 = net.predict(x)

    # in-memory arch + weights
    arch = json.loads(json.dumps(net.export_arch()))
    w = net.get_weights()
    s = hc.summarize_arch(net.dim, net.num_outputs, net.input_channels, net.layers)
    if s.total != net.weight_count:
        print(f"ERROR: summarize {s.total} != weight_count {net.weight_count}", file=sys.stderr)
        return 1

    net_b = hc.HCNN.from_arch(arch, num_threads=1, weight_seed=99)
    net_b.set_weights(w)
    if not np.array_equal(logits0, net_b.predict(x)):
        print("ERROR: set_weights logits mismatch", file=sys.stderr)
        return 1

    # file pair
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "model"
        net.save(stem)
        net_c = hc.HCNN.load(stem, num_threads=1)
        if not np.array_equal(logits0, net_c.predict(x)):
            print("ERROR: HCNN.load logits mismatch", file=sys.stderr)
            return 1

    max_abs = float(np.max(np.abs(logits0 - net_b.predict(x))))
    print(f"arch_and_weights_io: max_abs_logit_diff={max_abs:.3e}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
