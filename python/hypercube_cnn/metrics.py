"""Optional train-loop helpers: cosine LR and batch evaluate metrics.

Thin NumPy façade over C++ ``HCNNTrainHelpers`` (same definitions as the C++
MNIST / timeseries demos). Not part of the conv/pool graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._core import ClassEval, RegEval, cosine_lr as _cosine_lr
from ._core import evaluate_classification as _eval_cls
from ._core import evaluate_regression as _eval_reg

if TYPE_CHECKING:
    from . import HCNN

__all__ = [
    "ClassEval",
    "RegEval",
    "cosine_lr",
    "evaluate_classification",
    "evaluate_regression",
]


def cosine_lr(
    lr_max: float, lr_min: float, epoch: int, num_epochs: int
) -> float:
    """Cosine annealing: epoch 0 → ``lr_max``, last epoch → ``lr_min``.

    Matches ``hcnn::cosine_lr``. Typical MNIST: ``1e-3`` → ``1e-4``.
    """
    return float(_cosine_lr(float(lr_max), float(lr_min), int(epoch), int(num_epochs)))


def _batch_layout(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 3:
        b, c, n = arr.shape
        arr = arr.reshape(b, c * n)
    if arr.ndim != 2:
        raise ValueError("x must be 2D (batch, input_length) or 3D (batch, C, N)")
    return arr, int(arr.shape[0]), int(arr.shape[1])


def evaluate_classification(
    net: "HCNN", x: np.ndarray, targets: np.ndarray
) -> ClassEval:
    """Mean softmax CE and accuracy **percent** [0, 100] over a dataset.

    Parameters
    ----------
    x :
        ``(batch, input_length)`` or ``(batch, C, N)``. Prefer full capacity
        after spatial embed.
    targets :
        Class indices, length ``batch``.
    """
    arr, count, input_length = _batch_layout(x)
    y = np.ascontiguousarray(np.ravel(targets), dtype=np.int32)
    if y.size != count:
        raise ValueError("targets length must equal batch size")
    flat = np.ascontiguousarray(arr.reshape(-1))
    return _eval_cls(net._impl, flat, input_length, y, count)


def evaluate_regression(
    net: "HCNN", x: np.ndarray, targets: np.ndarray
) -> RegEval:
    """Mean MSE over all output dims (+ target variance / R²).

    Parameters
    ----------
    x :
        ``(batch, input_length)`` or ``(batch, C, N)``.
    targets :
        ``(batch, num_outputs)`` or flat ``batch * num_outputs``.
    """
    arr, count, input_length = _batch_layout(x)
    t = np.ascontiguousarray(targets, dtype=np.float32)
    if t.ndim == 2:
        if t.shape[0] != count or t.shape[1] != net.num_outputs:
            raise ValueError("targets shape must be (batch, num_outputs)")
        t = np.ascontiguousarray(t.reshape(-1))
    elif t.ndim == 1:
        if t.size != count * net.num_outputs:
            raise ValueError(
                "flat targets length must be batch * num_outputs"
            )
    else:
        raise ValueError("targets ndim must be 1 or 2")
    flat = np.ascontiguousarray(arr.reshape(-1))
    return _eval_reg(net._impl, flat, input_length, t, count, 0)
