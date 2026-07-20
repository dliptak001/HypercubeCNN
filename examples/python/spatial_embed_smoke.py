#!/usr/bin/env python3
"""Tier 1: spatial embed pad contract + DualPlane layout (no dataset download)."""

from __future__ import annotations

import sys

import numpy as np

import hypercube_cnn as hc


def main() -> int:
    # RowMajorPad must keep pad_value on the unused tail
    emb = hc.SpatialEmbedder(
        dim=6, mode=hc.SpatialEmbedMode.RowMajorPad, pad_value=-1.0
    )
    img = np.full((4, 4), 0.5, dtype=np.float32)
    out = emb.embed(img)
    if out.shape != (64,):
        print(f"ERROR: expected (64,), got {out.shape}", file=sys.stderr)
        return 1
    if not np.allclose(out[:16], 0.5) or not np.allclose(out[16:], -1.0):
        print("ERROR: RowMajorPad pad_value contract broken", file=sys.stderr)
        return 1

    # DualPlane on dim=9: N=512, S=16, pattern 2*S*S == N
    dual = hc.SpatialEmbedder(
        dim=9, mode=hc.SpatialEmbedMode.DualPlaneResize, pad_value=-1.0
    )
    digit = np.zeros((28, 28), dtype=np.float32)
    digit[8:20, 8:20] = 1.0
    packed = dual.embed(digit)
    plan = dual.plan(28, 28)
    if packed.shape != (512,) or plan.pattern_length != 512:
        print(
            f"ERROR: DualPlane layout N={packed.shape} pattern={plan.pattern_length}",
            file=sys.stderr,
        )
        return 1

    # Full-capacity path into a tiny net
    net = hc.HCNNConfig(
        dim=6,
        num_outputs=2,
        num_threads=1,
        layers=[hc.LayerSpec.conv(4)],
        weight_seed=1,
    ).build()
    x = emb.embed(img)
    logits = net.predict(x)
    if logits.shape != (2,):
        print(f"ERROR: predict after embed shape {logits.shape}", file=sys.stderr)
        return 1

    # Metrics + cosine endpoints (float32 C++ path; use relative tol)
    lr0 = hc.cosine_lr(1e-3, 1e-4, 0, 5)
    if abs(lr0 - 1e-3) > 1e-9:
        print(f"ERROR: cosine_lr epoch 0 got {lr0!r}", file=sys.stderr)
        return 1
    X = np.stack([x, x], axis=0)
    y = np.array([0, 1], dtype=np.int32)
    r = hc.evaluate_classification(net, X, y)
    if r.count != 2:
        print(f"ERROR: eval count {r.count}", file=sys.stderr)
        return 1

    print(
        f"spatial_embed_smoke: N={emb.N} dual_N={dual.N} "
        f"eval_acc={r.accuracy:.1f}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
