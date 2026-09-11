#!/usr/bin/env python3
"""Tier 1: spatial embed modes + pad contract (no dataset download)."""

from __future__ import annotations

import sys

import numpy as np

import hypercube_cnn as hc


def main() -> int:
    # PadLow must keep pad_value on the unused tail
    emb = hc.SpatialEmbedder(
        dim=6, mode=hc.SpatialEmbedMode.PadLow, pad_value=-1.0
    )
    img = np.full((4, 4), 0.5, dtype=np.float32)
    out = emb.embed(img)
    if out.shape != (64,):
        print(f"ERROR: expected (64,), got {out.shape}", file=sys.stderr)
        return 1
    if not np.allclose(out[:16], 0.5) or not np.allclose(out[16:], -1.0):
        print("ERROR: PadLow pad_value contract broken", file=sys.stderr)
        return 1

    # PadLowCenter: MNIST-shaped 28x28 @ dim=10 → 15x16 center, full N=1024
    center = hc.SpatialEmbedder(
        dim=10, mode=hc.SpatialEmbedMode.PadLowCenter, pad_value=-1.0
    )
    plan_c = center.plan(28, 28)
    if (
        plan_c.crop_h != 15
        or plan_c.crop_w != 16
        or plan_c.crop_row0 != 6
        or plan_c.crop_col0 != 6
        or plan_c.pattern_length != 1024
        or plan_c.N != 1024
    ):
        print(
            f"ERROR: PadLowCenter plan unexpected: crop={plan_c.crop_h}x{plan_c.crop_w}"
            f"@({plan_c.crop_row0},{plan_c.crop_col0}) "
            f"pattern={plan_c.pattern_length} N={plan_c.N}",
            file=sys.stderr,
        )
        return 1
    img28 = np.arange(28 * 28, dtype=np.float32).reshape(28, 28)
    packed_c = center.embed(img28)
    if packed_c.shape != (1024,):
        print(f"ERROR: PadLowCenter shape {packed_c.shape}", file=sys.stderr)
        return 1
    if not np.allclose(packed_c[:784], img28.reshape(-1)):
        print("ERROR: PadLowCenter prefix != full image", file=sys.stderr)
        return 1
    crop = img28[6 : 6 + 15, 6 : 6 + 16].reshape(-1)
    if not np.allclose(packed_c[784:], crop):
        print("ERROR: PadLowCenter tail != center crop", file=sys.stderr)
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
        f"spatial_embed_smoke: PadLow N={emb.N} "
        f"PadLowCenter N={center.N} dual_N={dual.N} "
        f"eval_acc={r.accuracy:.1f}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
