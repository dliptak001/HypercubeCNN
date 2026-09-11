"""Spatial preprocess: 2D aug + embed onto length-N hypercube inputs.

Optional product (not part of the conv/pool graph). Same contracts as
``docs/spatial_preprocess.md``:

- After embed, train/infer with **full capacity** ``N = 2**dim`` (do not pass
  a short pattern length if ``pad_value != 0`` — network zero-pad would wipe it).
- Prefer **aug then embed** (never aug packed vertices).
- Helpers are **single-channel** (``input_channels=1``); multi-channel packing
  is custom.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._core import (
    SpatialEmbedMode,
    SpatialEmbedPlan,
    _SpatialAugmenter,
    _SpatialEmbedder,
)

__all__ = [
    "SpatialEmbedMode",
    "SpatialEmbedPlan",
    "SpatialEmbedder",
    "SpatialAugmenter",
]


def _as_hw(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    arr = np.ascontiguousarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D (H, W), got shape {arr.shape}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    return arr, h, w


def _as_bhw(images: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    arr = np.ascontiguousarray(images, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim != 3:
        raise ValueError(
            f"images must be (H, W) or (B, H, W), got shape {images.shape}"
        )
    b, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    return arr, b, h, w


class SpatialEmbedder:
    """Map single-channel H×W images into length-N vertex buffers (P ≤ N).

    Parameters
    ----------
    dim :
        Hypercube dimension; capacity ``N = 2**dim``.
    mode :
        ``PadLow`` (full image + pad), ``PadLowCenter`` (full + center crop
        in remainder), ``ResizeToFit`` (square bilinear), or
        ``DualPlaneResize`` (ink ‖ max-norm |grad|). See
        ``docs/spatial_preprocess.md`` and ``docs/Python_SDK.md``.
    pad_value :
        Fill for unused vertices / bilinear OOB. Use ``-1`` for digit-like
        ``[-1, 1]`` ink (background), not the default ``0``.
    plane_side :
        Optional square side override for resize modes (``0`` = automatic).
        Ignored for ``PadLow`` / ``PadLowCenter``.

    Notes
    -----
    ``plan(H, W)`` returns a ``SpatialEmbedPlan``. For ``PadLowCenter`` the
    plan fills ``crop_h``, ``crop_w``, ``crop_row0``, ``crop_col0``; those
    fields are zero for other modes. Always train/infer with full capacity
    ``N`` so a non-zero ``pad_value`` is not wiped by network zero-pad.
    """

    def __init__(
        self,
        dim: int = 10,
        mode: SpatialEmbedMode = SpatialEmbedMode.PadLow,
        pad_value: float = 0.0,
        plane_side: int = 0,
    ):
        if not isinstance(mode, SpatialEmbedMode):
            raise TypeError(
                f"mode must be SpatialEmbedMode, got {type(mode).__name__}"
            )
        self._impl = _SpatialEmbedder(
            int(dim), mode, float(pad_value), int(plane_side)
        )

    @property
    def capacity(self) -> int:
        """``N = 2**dim`` — always the embed output length."""
        return int(self._impl.capacity)

    @property
    def dim(self) -> int:
        return int(self._impl.dim)

    @property
    def N(self) -> int:
        return self.capacity

    @property
    def mode(self) -> SpatialEmbedMode:
        return self._impl.mode

    @property
    def pad_value(self) -> float:
        return float(self._impl.pad_value)

    @property
    def plane_side(self) -> int:
        return int(self._impl.plane_side)

    def plan(self, height: int, width: int) -> SpatialEmbedPlan:
        """Describe layout for this input size (raises if invalid for mode)."""
        return self._impl.plan(int(height), int(width))

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Embed one ``(H, W)`` image → ``(N,)`` float32 full-capacity vector."""
        arr, h, w = _as_hw(image)
        return self._impl.embed(arr.ravel(), h, w)

    def embed_batch(self, images: np.ndarray) -> np.ndarray:
        """Embed ``(B, H, W)`` or ``(H, W)`` → ``(B, N)`` float32."""
        arr, b, h, w = _as_bhw(images)
        flat = np.ascontiguousarray(arr.reshape(-1))
        return self._impl.embed_batch(flat, b, h, w)

    @staticmethod
    def max_square_side(N: int) -> int:
        return int(_SpatialEmbedder.max_square_side(int(N)))

    @staticmethod
    def max_dual_plane_side(N: int) -> int:
        return int(_SpatialEmbedder.max_dual_plane_side(int(N)))

    def __repr__(self) -> str:
        return (
            f"SpatialEmbedder(dim={self.dim}, N={self.N}, "
            f"mode={self.mode.name}, pad_value={self.pad_value})"
        )


class SpatialAugmenter:
    """Optional 2D geometry / noise on H×W grids (DIM-agnostic).

    Apply **before** :class:`SpatialEmbedder`. Defaults are identity (no warp)
    when all geometry knobs are off; use ``enabled=False`` for a pure copy.
    """

    def __init__(
        self,
        *,
        rot_deg_max: float = 0.0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        shift_max: int = 0,
        shear_x_max: float = 0.0,
        shear_y_max: float = 0.0,
        elastic_alpha: float = 0.0,
        elastic_sigma: float = 0.0,
        noise_sigma: float = 0.0,
        value_min: float = -1.0,
        value_max: float = 1.0,
        border_value: float = 0.0,
        enabled: bool = True,
    ):
        self._impl = _SpatialAugmenter(
            float(rot_deg_max),
            float(scale_min),
            float(scale_max),
            int(shift_max),
            float(shear_x_max),
            float(shear_y_max),
            float(elastic_alpha),
            float(elastic_sigma),
            float(noise_sigma),
            float(value_min),
            float(value_max),
            float(border_value),
            bool(enabled),
        )

    def apply(self, image: np.ndarray, *, seed: int = 0) -> np.ndarray:
        """Augment one ``(H, W)`` image → ``(H, W)`` float32."""
        arr, h, w = _as_hw(image)
        out = self._impl.apply(arr.ravel(), h, w, int(seed))
        return np.asarray(out, dtype=np.float32).reshape(h, w)

    def apply_batch(self, images: np.ndarray, *, seed: int = 0) -> np.ndarray:
        """Augment ``(B, H, W)`` → ``(B, H, W)`` float32."""
        arr, b, h, w = _as_bhw(images)
        flat = np.ascontiguousarray(arr.reshape(-1))
        out = self._impl.apply_batch(flat, b, h, w, int(seed))
        return np.asarray(out, dtype=np.float32).reshape(b, h, w)

    def __repr__(self) -> str:
        return "SpatialAugmenter(...)"
