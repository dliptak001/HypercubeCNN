"""Architecture product surface: LayerSpec, summarize, HCNNConfig, arch JSON.

Pure Python — mirrors ``HCNNArch.h`` contracts without binding the header.
JSON sidecar format is versioned (``format`` / ``version``); unknown future
versions are rejected with an upgrade message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, TYPE_CHECKING, Union

from ._core import Activation, OptimizerType, PoolType, TaskType

if TYPE_CHECKING:
    from . import HCNN

# ── Arch JSON format ──

ARCH_FORMAT = "hcnn_arch"
ARCH_VERSION = 1  # bump when sidecar fields change incompatibly

_ACTIVATION_TO_STR = {
    Activation.NONE: "none",
    Activation.RELU: "relu",
    Activation.LEAKY_RELU: "leaky_relu",
    Activation.TANH: "tanh",
}
_STR_TO_ACTIVATION = {v: k for k, v in _ACTIVATION_TO_STR.items()}
# accept C++-style names too (pybind11 enums are not iterable)
for _act, _name in (
    (Activation.NONE, "NONE"),
    (Activation.RELU, "RELU"),
    (Activation.LEAKY_RELU, "LEAKY_RELU"),
    (Activation.TANH, "TANH"),
):
    _STR_TO_ACTIVATION[_name] = _act
    _STR_TO_ACTIVATION[_name.lower()] = _act

_POOL_TO_STR = {PoolType.MAX: "max", PoolType.AVG: "avg"}
_STR_TO_POOL = {
    "max": PoolType.MAX,
    "avg": PoolType.AVG,
    "MAX": PoolType.MAX,
    "AVG": PoolType.AVG,
}

_TASK_TO_STR = {
    TaskType.Classification: "classification",
    TaskType.Regression: "regression",
}
_STR_TO_TASK = {
    "classification": TaskType.Classification,
    "regression": TaskType.Regression,
    "Classification": TaskType.Classification,
    "Regression": TaskType.Regression,
}


def _parse_activation(value: Union[Activation, str]) -> Activation:
    if isinstance(value, Activation):
        return value
    key = str(value).strip()
    if key not in _STR_TO_ACTIVATION:
        raise ValueError(
            f"activation must be one of {sorted(set(_ACTIVATION_TO_STR.values()))}, "
            f"got {value!r}"
        )
    return _STR_TO_ACTIVATION[key]


def _parse_pool(value: Union[PoolType, str]) -> PoolType:
    if isinstance(value, PoolType):
        return value
    if value not in _STR_TO_POOL:
        raise ValueError(f"pool_type must be 'max' or 'avg', got {value!r}")
    return _STR_TO_POOL[value]


def _parse_task(value: Union[TaskType, str]) -> TaskType:
    if isinstance(value, TaskType):
        return value
    if value not in _STR_TO_TASK:
        raise ValueError(
            f"task must be 'classification' or 'regression', got {value!r}"
        )
    return _STR_TO_TASK[value]


@dataclass
class LayerSpec:
    """One body step: Hamming conv or antipodal pool (C++ ``hcnn::LayerSpec``)."""

    kind: str  # "conv" | "pool"
    c_out: int = 16
    activation: Activation = Activation.RELU
    use_bias: bool = True
    use_bn: bool = False
    pool_type: PoolType = PoolType.MAX

    @staticmethod
    def conv(
        c_out: int,
        activation: Union[Activation, str] = Activation.RELU,
        use_bias: bool = True,
        bn: bool = False,
    ) -> "LayerSpec":
        if not isinstance(c_out, int) or c_out < 1:
            raise ValueError(f"c_out must be a positive int, got {c_out!r}")
        return LayerSpec(
            kind="conv",
            c_out=c_out,
            activation=_parse_activation(activation),
            use_bias=bool(use_bias),
            use_bn=bool(bn),
        )

    @staticmethod
    def pool(pool_type: Union[PoolType, str] = PoolType.MAX) -> "LayerSpec":
        return LayerSpec(kind="pool", pool_type=_parse_pool(pool_type))

    def to_dict(self) -> dict:
        if self.kind == "conv":
            return {
                "kind": "conv",
                "c_out": int(self.c_out),
                "activation": _ACTIVATION_TO_STR[self.activation],
                "use_bias": bool(self.use_bias),
                "use_bn": bool(self.use_bn),
            }
        if self.kind == "pool":
            return {
                "kind": "pool",
                "pool_type": _POOL_TO_STR[self.pool_type],
            }
        raise ValueError(f"unknown layer kind {self.kind!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "LayerSpec":
        if not isinstance(d, dict):
            raise TypeError(f"layer must be a dict, got {type(d).__name__}")
        kind = d.get("kind")
        if kind == "conv":
            return cls.conv(
                int(d["c_out"]),
                activation=d.get("activation", "relu"),
                use_bias=bool(d.get("use_bias", True)),
                bn=bool(d.get("use_bn", False)),
            )
        if kind == "pool":
            return cls.pool(d.get("pool_type", "max"))
        raise ValueError(f"layer kind must be 'conv' or 'pool', got {kind!r}")


@dataclass
class ArchParamSummary:
    """Parameter accounting matching ``HCNN.weight_count`` after randomize."""

    total: int = 0
    readout: int = 0
    flatten_features: int = 0
    final_dim: int = 0
    final_N: int = 0
    last_channels: int = 0
    num_conv: int = 0
    num_pool: int = 0
    conv_params: List[int] = field(default_factory=list)


def summarize_arch(
    dim: int,
    num_outputs: int,
    input_channels: int,
    layers: Sequence[LayerSpec],
) -> ArchParamSummary:
    """Walk a layer list; total matches ``GetWeightCount`` after randomize.

    Same validation rules as C++ ``summarize_arch`` (dim [3,30], ≥1 conv, etc.).
    """
    if not (3 <= dim <= 30):
        raise ValueError(f"summarize_arch: dim must be in [3, 30], got {dim}")
    if num_outputs < 1:
        raise ValueError("summarize_arch: num_outputs must be >= 1")
    if input_channels < 1:
        raise ValueError("summarize_arch: input_channels must be >= 1")
    if not layers:
        raise ValueError("summarize_arch: need at least one layer")

    s = ArchParamSummary()
    d = dim
    N = 1 << d
    c_in = input_channels
    s.last_channels = c_in

    for L in layers:
        if not isinstance(L, LayerSpec):
            raise TypeError(f"layers must be LayerSpec, got {type(L).__name__}")
        if L.kind == "conv":
            if L.c_out < 1:
                raise ValueError("summarize_arch: conv c_out must be >= 1")
            k_params = c_in * L.c_out * (d + 1) + (L.c_out if L.use_bias else 0)
            if L.use_bn:
                k_params += 4 * L.c_out
            s.conv_params.append(int(k_params))
            s.total += k_params
            c_in = L.c_out
            s.last_channels = L.c_out
            s.num_conv += 1
        elif L.kind == "pool":
            if d < 2:
                raise ValueError(
                    f"summarize_arch: cannot pool at current_dim={d} (need >= 2)"
                )
            d -= 1
            N = 1 << d
            s.num_pool += 1
        else:
            raise ValueError(f"summarize_arch: unknown kind {L.kind!r}")

    if s.num_conv < 1:
        raise ValueError("summarize_arch: need at least one Conv layer")

    s.final_dim = d
    s.final_N = N
    s.flatten_features = s.last_channels * N
    s.readout = s.flatten_features * num_outputs + num_outputs
    s.total += s.readout
    return s


def apply_arch(net: "HCNN", layers: Sequence[LayerSpec]) -> ArchParamSummary:
    """Append ``layers`` onto ``net``. Validates the **full** stack first.

    Unlike a naive summarize of only the new layers (which assumes an empty
    body at ``net.dim``), this accounts for layers already on ``net`` so the
    returned :class:`ArchParamSummary` matches ``weight_count`` after
    ``randomize_weights``. Prefer applying onto an empty body for clarity.
    """
    layers = layers_from_iterable(layers)
    combined = list(net.layers) + list(layers)
    summary = summarize_arch(
        net.dim, net.num_outputs, net.input_channels, combined
    )
    for L in layers:
        if L.kind == "conv":
            net.add_conv(
                L.c_out,
                activation=L.activation,
                use_bias=L.use_bias,
                use_bn=L.use_bn,
            )
        else:
            net.add_pool(L.pool_type)
    return summary


@dataclass
class HCNNConfig:
    """One-shot build knobs (C++ ``hcnn::HCNNConfig``).

    ``build()`` order: construct → apply layers → optional randomize → set optimizer.
    Default optimizer is Adam (same as C++).
    """

    dim: int = 10
    num_outputs: int = 10
    input_channels: int = 1
    task: Union[TaskType, str] = TaskType.Classification
    num_threads: int = 0
    layers: List[LayerSpec] = field(default_factory=list)

    optimizer: OptimizerType = OptimizerType.ADAM
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    randomize: bool = True
    weight_scale: float = 0.0
    # Full 64-bit weight-init master seed (C++ uint64_t); validated in build().
    weight_seed: int = 42

    def summarize(self) -> ArchParamSummary:
        return summarize_arch(
            self.dim, self.num_outputs, self.input_channels, self.layers
        )

    def build(self) -> "HCNN":
        # Local import avoids circular init
        from . import HCNN

        task = _parse_task(self.task)
        self.summarize()  # validate early
        net = HCNN(
            dim=self.dim,
            num_outputs=self.num_outputs,
            input_channels=self.input_channels,
            task=task,
            num_threads=self.num_threads,
        )
        apply_arch(net, self.layers)
        if self.randomize:
            # randomize_weights validates weight_seed as full uint64.
            net.randomize_weights(self.weight_scale, self.weight_seed)
        net.set_optimizer(
            self.optimizer, self.adam_beta1, self.adam_beta2, self.adam_eps
        )
        return net

    def to_arch_dict(self) -> dict:
        """JSON-serializable sidecar (no optimizer / train session state)."""
        task = _parse_task(self.task)
        return {
            "format": ARCH_FORMAT,
            "version": ARCH_VERSION,
            "dim": int(self.dim),
            "num_outputs": int(self.num_outputs),
            "input_channels": int(self.input_channels),
            "task": _TASK_TO_STR[task],
            "layers": [L.to_dict() for L in self.layers],
        }

    @classmethod
    def from_arch_dict(cls, arch: dict) -> "HCNNConfig":
        """Parse arch sidecar; reject unknown future format versions."""
        if not isinstance(arch, dict):
            raise TypeError(f"arch must be a dict, got {type(arch).__name__}")
        fmt = arch.get("format", ARCH_FORMAT)
        if fmt != ARCH_FORMAT:
            raise ValueError(
                f"Unknown arch format {fmt!r} (expected {ARCH_FORMAT!r})"
            )
        version = int(arch.get("version", 0))
        if version > ARCH_VERSION:
            raise ValueError(
                f"Arch was saved with format version {version}, but this "
                f"library only supports up to {ARCH_VERSION}. Upgrade hypercube-cnn."
            )
        if version < 1:
            raise ValueError(f"Invalid arch format version {version}")
        layers_raw = arch.get("layers")
        if not isinstance(layers_raw, list):
            raise ValueError("arch['layers'] must be a list")
        layers = [LayerSpec.from_dict(L) for L in layers_raw]
        return cls(
            dim=int(arch["dim"]),
            num_outputs=int(arch["num_outputs"]),
            input_channels=int(arch.get("input_channels", 1)),
            task=_parse_task(arch.get("task", "classification")),
            layers=layers,
        )


def layers_from_iterable(layers: Iterable[Any]) -> List[LayerSpec]:
    """Normalize a sequence of LayerSpec or dicts."""
    out: List[LayerSpec] = []
    for L in layers:
        if isinstance(L, LayerSpec):
            out.append(L)
        elif isinstance(L, dict):
            out.append(LayerSpec.from_dict(L))
        else:
            raise TypeError(
                f"layer must be LayerSpec or dict, got {type(L).__name__}"
            )
    return out
