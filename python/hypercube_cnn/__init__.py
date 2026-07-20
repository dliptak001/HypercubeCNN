"""HypercubeCNN: dependency-free hypercube CNN core.

Python bindings for the C++ ``HCNN`` front door. Capacity is topological:
per channel the network holds ``N = 2**dim`` vertices. Pack non-power-of-two
data in the host before calling train/infer.

This class is **not thread-safe**. Do not share one instance across threads.
The GIL is released during train/infer C++ calls; the same network must not
be accessed concurrently.

See ``docs/python_sdk_plan.md`` and ``docs/CPP_SDK.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from ._core import (
    Activation,
    OptimizerType,
    PoolType,
    TaskType,
    _HCNN,
    __version__ as _core_version,
)
from .arch import (
    ARCH_FORMAT,
    ARCH_VERSION,
    ArchParamSummary,
    HCNNConfig,
    LayerSpec,
    apply_arch,
    layers_from_iterable,
    summarize_arch,
)

__version__ = _core_version
__all__ = [
    "HCNN",
    "TrainParams",
    "LayerSpec",
    "HCNNConfig",
    "ArchParamSummary",
    "summarize_arch",
    "apply_arch",
    "Activation",
    "PoolType",
    "TaskType",
    "OptimizerType",
    "ARCH_FORMAT",
    "ARCH_VERSION",
    "__version__",
]


def _to_float32(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def _to_int32(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


@dataclass
class TrainParams:
    """Session / per-call training knobs (maps to C++ ``hcnn::TrainParams``).

    Parameters
    ----------
    learning_rate :
        Step size. Default ``1e-3`` (Adam-friendly).
    momentum :
        SGD momentum; ignored by Adam.
    weight_decay :
        L2 / AdamW-style decoupled decay on kernels.
    shuffle_seed :
        Epoch only: ``0`` = sequential; nonzero = deterministic shuffle.
    class_weights :
        Classification only: optional per-class loss scale, length
        ``num_outputs``, float32.
    """

    learning_rate: float = 1e-3
    momentum: float = 0.0
    weight_decay: float = 0.0
    shuffle_seed: int = 0
    class_weights: Optional[np.ndarray] = None


class HCNN:
    """Hypercube CNN front door (integration surface).

    Build a stack with :meth:`add_conv` / :meth:`add_pool`, then
    :meth:`randomize_weights` before train or infer. Changing the stack after
    randomization **invalidates** weights — call :meth:`randomize_weights`
    again.

    Parameters
    ----------
    dim :
        Hypercube dimension in ``[3, 30]``. Capacity per channel is ``N = 2**dim``.
    num_outputs :
        Logit / regression width.
    input_channels :
        Input channels (channel-major layout).
    task :
        :class:`TaskType.Classification` (CE) or :class:`TaskType.Regression`
        (sum-style MSE). Fixed at construction.
    num_threads :
        ``0`` = auto, ``1`` = single-threaded, ``N`` = N workers.
        Use ``1`` when the host parallelizes across many nets.
    """

    def __init__(
        self,
        dim: int,
        num_outputs: int = 10,
        input_channels: int = 1,
        task: TaskType = TaskType.Classification,
        num_threads: int = 0,
    ):
        if not isinstance(dim, int) or not (3 <= dim <= 30):
            raise ValueError(f"dim must be an integer in [3, 30], got {dim!r}")
        if not isinstance(num_outputs, int) or num_outputs < 1:
            raise ValueError(f"num_outputs must be a positive int, got {num_outputs!r}")
        if not isinstance(input_channels, int) or input_channels < 1:
            raise ValueError(
                f"input_channels must be a positive int, got {input_channels!r}"
            )
        if not isinstance(num_threads, int) or num_threads < 0:
            raise ValueError(
                f"num_threads must be a non-negative int, got {num_threads!r}"
            )
        if not isinstance(task, TaskType):
            raise TypeError(f"task must be TaskType, got {type(task).__name__}")
        self._impl = _HCNN(
            start_dim=dim,
            num_outputs=num_outputs,
            input_channels=input_channels,
            task_type=task,
            num_threads=num_threads,
        )
        # Recorded for export_arch / arch sidecar (not stored in C++ weights).
        self._layers: List[LayerSpec] = []
        # Python-side session defaults (bindings always pass explicit TrainParams
        # fields; keep a copy so train_* honor set_train_defaults when params=None).
        self._train_defaults = TrainParams()

    # ── Architecture ──

    def add_conv(
        self,
        c_out: int,
        activation: Activation = Activation.RELU,
        use_bias: bool = True,
        use_bn: bool = False,
    ) -> None:
        """Append a Hamming conv layer. Invalidates weights if already randomized."""
        if not isinstance(c_out, int) or c_out < 1:
            raise ValueError(f"c_out must be a positive int, got {c_out!r}")
        if not isinstance(activation, Activation):
            raise TypeError(
                f"activation must be Activation, got {type(activation).__name__}"
            )
        self._impl.add_conv(c_out, activation, use_bias, use_bn)
        self._layers.append(
            LayerSpec.conv(c_out, activation=activation, use_bias=use_bias, bn=use_bn)
        )

    def add_pool(self, pool_type: PoolType = PoolType.MAX) -> None:
        """Append an antipodal pool (DIM -= 1). Invalidates weights if randomized."""
        if not isinstance(pool_type, PoolType):
            raise TypeError(f"pool_type must be PoolType, got {type(pool_type).__name__}")
        self._impl.add_pool(pool_type)
        self._layers.append(LayerSpec.pool(pool_type))

    def apply_layers(self, layers: Sequence[LayerSpec]) -> ArchParamSummary:
        """Append a list of :class:`LayerSpec` (validates with :func:`summarize_arch`)."""
        return apply_arch(self, layers_from_iterable(layers))

    @property
    def layers(self) -> List[LayerSpec]:
        """Copy of recorded body layers (for arch export)."""
        return list(self._layers)

    def export_arch(self) -> dict:
        """JSON-serializable arch sidecar (not weights).

        Rebuild with :meth:`from_arch`, then :meth:`set_weights` / load HCNW.
        Requires at least one recorded layer (via :meth:`add_conv` / :meth:`apply_layers`).
        """
        if not self._layers:
            raise ValueError(
                "export_arch: no layers recorded; add_conv/add_pool or build from HCNNConfig"
            )
        cfg = HCNNConfig(
            dim=self.dim,
            num_outputs=self.num_outputs,
            input_channels=self.input_channels,
            task=self.task,
            layers=list(self._layers),
        )
        return cfg.to_arch_dict()

    @classmethod
    def from_arch(
        cls,
        arch: dict,
        *,
        num_threads: Optional[int] = None,
        randomize: bool = True,
        weight_scale: float = 0.0,
        weight_seed: int = 42,
        optimizer: OptimizerType = OptimizerType.ADAM,
    ) -> "HCNN":
        """Rebuild a net from :meth:`export_arch` / arch JSON.

        Default randomizes weights (required before :meth:`set_weights`).
        Optimizer defaults to Adam (fresh Build semantics), not stored in the sidecar.
        """
        cfg = HCNNConfig.from_arch_dict(arch)
        if num_threads is not None:
            cfg.num_threads = int(num_threads)
        cfg.randomize = bool(randomize)
        cfg.weight_scale = float(weight_scale)
        cfg.weight_seed = int(weight_seed)
        cfg.optimizer = optimizer
        return cfg.build()

    @classmethod
    def from_layers(
        cls,
        layers: Sequence[Union[LayerSpec, dict]],
        *,
        dim: int = 10,
        num_outputs: int = 10,
        input_channels: int = 1,
        task: TaskType = TaskType.Classification,
        num_threads: int = 0,
        randomize: bool = True,
        weight_seed: int = 42,
    ) -> "HCNN":
        """One-shot construct from a layer list (see also :class:`HCNNConfig`)."""
        cfg = HCNNConfig(
            dim=dim,
            num_outputs=num_outputs,
            input_channels=input_channels,
            task=task,
            num_threads=num_threads,
            layers=layers_from_iterable(layers),
            randomize=randomize,
            weight_seed=weight_seed,
        )
        return cfg.build()

    def randomize_weights(self, scale: float = 0.0, seed: int = 42) -> None:
        """Initialize weights for the current stack. Required before train/infer.

        ``scale > 0``: uniform ``[-scale, +scale]``; ``scale <= 0``: Xavier/He.
        """
        self._impl.randomize_weights(float(scale), int(seed))

    # ── Mode / optimizer ──

    def set_training(self, training: bool) -> None:
        """Train vs eval mode (matters when batch-norm is enabled)."""
        self._impl.set_training(bool(training))

    def set_optimizer(
        self,
        opt: OptimizerType,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Configure optimizer. Default at construction is Adam."""
        if not isinstance(opt, OptimizerType):
            raise TypeError(f"opt must be OptimizerType, got {type(opt).__name__}")
        self._impl.set_optimizer(opt, float(beta1), float(beta2), float(eps))

    def set_train_defaults(self, params: TrainParams) -> None:
        """Session defaults for train calls that omit ``params``.

        Stores a copy on this instance (including optional ``class_weights``)
        and mirrors the numeric fields into the C++ session defaults.
        """
        if not isinstance(params, TrainParams):
            raise TypeError("params must be TrainParams")
        cw = None
        if params.class_weights is not None:
            cw = _to_float32(np.ravel(params.class_weights)).copy()
        self._train_defaults = TrainParams(
            learning_rate=float(params.learning_rate),
            momentum=float(params.momentum),
            weight_decay=float(params.weight_decay),
            shuffle_seed=int(params.shuffle_seed),
            class_weights=cw,
        )
        self._impl.set_train_defaults(
            float(params.learning_rate),
            float(params.momentum),
            float(params.weight_decay),
            int(params.shuffle_seed),
        )

    def prepare_buffers(self) -> None:
        """Eagerly allocate work buffers (optional; after randomize_weights)."""
        self._impl.prepare_buffers()

    # ── Inference ──

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Embed + forward. Returns ``(num_outputs,)`` float32 logits/preds.

        ``x`` is flattened to a contiguous float32 vector (length may be short;
        C++ zero-pads to capacity). Prefer full capacity after host packing.
        """
        return self._impl.predict(_to_float32(np.ravel(x)))

    def predict_class(self, x: np.ndarray) -> int:
        """Classification only: argmax of :meth:`predict`."""
        return int(self._impl.predict_class(_to_float32(np.ravel(x))))

    def forward(self, embedded: np.ndarray) -> np.ndarray:
        """Forward from already-embedded activations of length ``C * N``."""
        return self._impl.forward(_to_float32(np.ravel(embedded)))

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        """Batch inference.

        Parameters
        ----------
        x :
            Shape ``(batch, input_length)`` or ``(batch, C, N)`` (flattened per
            sample). Returns ``(batch, num_outputs)``.
        """
        arr = _to_float32(x)
        if arr.ndim == 1:
            raise ValueError(
                "forward_batch expects 2D (batch, input_length) or 3D (batch, C, N)"
            )
        if arr.ndim == 3:
            b, c, n = arr.shape
            arr = arr.reshape(b, c * n)
        if arr.ndim != 2:
            raise ValueError(f"forward_batch: unsupported ndim {arr.ndim}")
        batch_size, input_length = int(arr.shape[0]), int(arr.shape[1])
        flat = np.ascontiguousarray(arr.reshape(-1))
        return self._impl.forward_batch(flat, input_length, batch_size)

    # ── Training ──

    def _params_fields(self, params: Optional[TrainParams]):
        p = self._train_defaults if params is None else params
        cw = None
        if p.class_weights is not None:
            cw = _to_float32(np.ravel(p.class_weights))
        return (
            float(p.learning_rate),
            float(p.momentum),
            float(p.weight_decay),
            int(p.shuffle_seed),
            cw,
        )

    def train_step(
        self,
        x: np.ndarray,
        target: Union[int, np.ndarray],
        params: Optional[TrainParams] = None,
    ) -> None:
        """One sample gradient step.

        Classification: ``target`` is a class index (int).
        Regression: ``target`` is a length-``num_outputs`` float array.
        """
        lr, mom, wd, _, cw = self._params_fields(params)
        xin = _to_float32(np.ravel(x))
        if self.task == TaskType.Classification:
            if isinstance(target, (bool, np.bool_)):
                raise TypeError("target class must be an integer")
            if isinstance(target, (int, np.integer)):
                tcls = int(target)
            else:
                tarr = np.asarray(target).ravel()
                if tarr.size != 1:
                    raise ValueError(
                        "classification target must be a single class index"
                    )
                tcls = int(tarr[0])
            self._impl.train_step_class(xin, tcls, lr, mom, wd, cw)
        else:
            t = _to_float32(np.ravel(target))
            self._impl.train_step_reg(xin, t, lr, mom, wd)

    def train_batch(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        params: Optional[TrainParams] = None,
    ) -> None:
        """One mini-batch gradient step.

        ``x``: ``(batch, input_length)`` or ``(batch, C, N)``.
        Classification targets: ``(batch,)`` int.
        Regression targets: ``(batch, num_outputs)`` float.
        """
        lr, mom, wd, _, cw = self._params_fields(params)
        arr = _to_float32(x)
        if arr.ndim == 3:
            b, c, n = arr.shape
            arr = arr.reshape(b, c * n)
        if arr.ndim != 2:
            raise ValueError("train_batch: x must be 2D or 3D")
        batch_size, input_length = int(arr.shape[0]), int(arr.shape[1])
        flat = np.ascontiguousarray(arr.reshape(-1))
        if self.task == TaskType.Classification:
            y = _to_int32(np.ravel(targets))
            if y.size != batch_size:
                raise ValueError("train_batch: targets length must equal batch size")
            self._impl.train_batch_class(
                flat, input_length, y, batch_size, lr, mom, wd, cw
            )
        else:
            t = _to_float32(targets)
            if t.ndim == 1:
                if t.size != batch_size * self.num_outputs:
                    raise ValueError(
                        "train_batch: flat targets length must be batch * num_outputs"
                    )
            elif t.ndim == 2:
                if t.shape[0] != batch_size or t.shape[1] != self.num_outputs:
                    raise ValueError(
                        "train_batch: targets shape must be (batch, num_outputs)"
                    )
                t = np.ascontiguousarray(t.reshape(-1))
            else:
                raise ValueError("train_batch: targets ndim must be 1 or 2")
            self._impl.train_batch_reg(
                flat, input_length, t, batch_size, lr, mom, wd
            )

    def train_epoch(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        params: Optional[TrainParams] = None,
    ) -> None:
        """One epoch over a dataset (optional shuffle via ``params.shuffle_seed``).

        ``x``: ``(samples, input_length)`` or ``(samples, C, N)``.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        lr, mom, wd, shuffle_seed, cw = self._params_fields(params)
        arr = _to_float32(x)
        if arr.ndim == 3:
            b, c, n = arr.shape
            arr = arr.reshape(b, c * n)
        if arr.ndim != 2:
            raise ValueError("train_epoch: x must be 2D or 3D")
        sample_count, input_length = int(arr.shape[0]), int(arr.shape[1])
        flat = np.ascontiguousarray(arr.reshape(-1))
        if self.task == TaskType.Classification:
            y = _to_int32(np.ravel(targets))
            if y.size != sample_count:
                raise ValueError("train_epoch: targets length must equal sample count")
            self._impl.train_epoch_class(
                flat,
                input_length,
                y,
                sample_count,
                batch_size,
                lr,
                mom,
                wd,
                shuffle_seed,
                cw,
            )
        else:
            t = _to_float32(targets)
            if t.ndim == 2:
                if t.shape[0] != sample_count or t.shape[1] != self.num_outputs:
                    raise ValueError(
                        "train_epoch: targets shape must be (samples, num_outputs)"
                    )
                t = np.ascontiguousarray(t.reshape(-1))
            elif t.ndim == 1:
                if t.size != sample_count * self.num_outputs:
                    raise ValueError(
                        "train_epoch: flat targets length must be samples * num_outputs"
                    )
            else:
                raise ValueError("train_epoch: targets ndim must be 1 or 2")
            self._impl.train_epoch_reg(
                flat,
                input_length,
                t,
                sample_count,
                batch_size,
                lr,
                mom,
                wd,
                shuffle_seed,
            )

    # ── Weights ──

    @property
    def weight_count(self) -> int:
        return int(self._impl.weight_count())

    def get_weights(self) -> np.ndarray:
        """Copy parameter blob as float32 (not optimizer moments)."""
        return self._impl.get_weights()

    def set_weights(
        self, data: np.ndarray, *, reset_optimizer_moments: bool = False
    ) -> None:
        """Restore from :meth:`get_weights` layout."""
        self._impl.set_weights(_to_float32(np.ravel(data)), reset_optimizer_moments)

    # ── HCNW + arch sidecar file I/O ──

    def save_weights(self, path: Union[str, Path]) -> None:
        """Write an HCNW parameter file (not the layer graph).

        Pair with :meth:`export_arch` JSON for a full restore. See :meth:`save`.
        """
        self._impl.save_weights(str(Path(path)))

    def load_weights(
        self, path: Union[str, Path], *, reset_optimizer_moments: bool = False
    ) -> None:
        """Load HCNW into this net (must already match the saved architecture).

        Requires :meth:`randomize_weights` first (or a build that randomized).
        Architecture mismatch, bad magic, or version errors raise ``RuntimeError``.
        """
        self._impl.load_weights(str(Path(path)), reset_optimizer_moments)

    @staticmethod
    def _model_paths(path: Union[str, Path]) -> tuple[Path, Path]:
        """Resolve ``stem.hcnw`` + ``stem.arch.json`` from a path or stem."""
        p = Path(path)
        name = p.name
        if name.endswith(".arch.json"):
            stem = name[: -len(".arch.json")]
            return p.parent / f"{stem}.hcnw", p
        if p.suffix.lower() == ".hcnw":
            return p, p.parent / f"{p.stem}.arch.json"
        # bare stem (or other suffix): append both extensions
        return Path(str(p) + ".hcnw"), Path(str(p) + ".arch.json")

    def save(self, path: Union[str, Path]) -> None:
        """Write ``path.hcnw`` + ``path.arch.json`` (or sibling pair if ``.hcnw``).

        HCNW stores parameters only. The arch JSON is required to rebuild the
        layer stack before :meth:`load_weights`.
        """
        hcnw, arch = self._model_paths(path)
        arch.parent.mkdir(parents=True, exist_ok=True)
        with open(arch, "w", encoding="utf-8") as f:
            json.dump(self.export_arch(), f, indent=2)
            f.write("\n")
        self.save_weights(hcnw)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        num_threads: Optional[int] = None,
        reset_optimizer_moments: bool = False,
    ) -> "HCNN":
        """Load a model saved by :meth:`save` (arch JSON + HCNW).

        Rebuilds from the arch sidecar, randomizes, then loads weights.
        Optimizer is Adam default (not stored in HCNW).
        """
        hcnw, arch = cls._model_paths(path)
        if not arch.is_file():
            raise FileNotFoundError(
                f"arch sidecar not found: {arch} "
                f"(expected beside {hcnw}; save with HCNN.save or write export_arch JSON)"
            )
        if not hcnw.is_file():
            raise FileNotFoundError(f"HCNW weights not found: {hcnw}")
        with open(arch, encoding="utf-8") as f:
            arch_dict = json.load(f)
        net = cls.from_arch(arch_dict, num_threads=num_threads, randomize=True)
        net.load_weights(hcnw, reset_optimizer_moments=reset_optimizer_moments)
        return net

    # ── Properties ──

    @property
    def dim(self) -> int:
        return int(self._impl.start_dim)

    @property
    def N(self) -> int:
        """Capacity vertices per channel: ``2**dim``."""
        return int(self._impl.start_n)

    @property
    def start_n(self) -> int:
        return int(self._impl.start_n)

    @property
    def current_dim(self) -> int:
        return int(self._impl.current_dim)

    @property
    def input_channels(self) -> int:
        return int(self._impl.input_channels)

    @property
    def num_outputs(self) -> int:
        return int(self._impl.num_outputs)

    @property
    def num_conv(self) -> int:
        return int(self._impl.num_conv)

    @property
    def num_pool(self) -> int:
        return int(self._impl.num_pool)

    @property
    def task(self) -> TaskType:
        return self._impl.task_type

    @property
    def optimizer(self) -> OptimizerType:
        return self._impl.optimizer_type

    @property
    def weights_initialized(self) -> bool:
        return bool(self._impl.weights_initialized)

    @property
    def capacity(self) -> int:
        """``input_channels * N`` floats per sample at full capacity."""
        return self.input_channels * self.N

    def __repr__(self) -> str:
        return (
            f"HCNN(dim={self.dim}, N={self.N}, channels={self.input_channels}, "
            f"outputs={self.num_outputs}, task={self.task.name}, "
            f"conv={self.num_conv}, pool={self.num_pool}, "
            f"weights_initialized={self.weights_initialized})"
        )
