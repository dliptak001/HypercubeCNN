# HypercubeCNN — Concept

**Author:** David Liptak
**Date:** April 2026 (revised)

> See [architecture.md](architecture.md) for the technical implementation, [CPP_SDK.md](CPP_SDK.md) for the C++ API, and [mnist.md](mnist.md) for benchmark results.

## What is HypercubeCNN?

HypercubeCNN is a convolutional neural network that operates on binary hypercubes instead of spatial grids. The substrate is a DIM-dimensional hypercube with N = 2^DIM vertices. Each vertex is a DIM-bit integer; neighbors are reached by flipping single bits (XOR). All geometry is bitwise — no adjacency lists, no spatial coordinates, no padding.

The convolution kernel learns one weight per bit-flip direction (per input/output channel pair), shared across all vertices. This is the hypercube analogue of a 3x3 spatial kernel shared across all pixel positions. The weight sharing is mathematically exact: the hypercube's vertex-transitive symmetry (the group Z_2^n) guarantees that every vertex has identical local structure.

## Where it came from

HypercubeCNN is the third project built on the binary hypercube primitive:

- **HypercubeRC** — reservoir computing on hypercube topology. Demonstrated XOR-based neighbor lookup, Hamming-shell connectivity, and scale-invariant hyperparameters.
- **HypercubeHopfield** — sparse local-attention Hopfield network. Demonstrated Hamming-ball neighborhoods for associative memory with >250x capacity scaling.

Both projects proved the hypercube is a practical, storage-free, hardware-native substrate for neural computation. HypercubeCNN asks: can the same substrate support learned convolutional feature extraction with end-to-end backpropagation?

**Important:** HRC and HHOP are conceptual predecessors only. There is no code integration between the projects.

## The core operation

For each output channel and each vertex v:

```
out(v) = bias + sum over (input_channel, k) of w[channel, k] * in[channel, v ^ (1 << k)]
```

The expression `v ^ (1 << k)` is an XOR that flips bit k of vertex v, yielding the nearest neighbor along dimension k. With K = DIM directions per vertex, the kernel sees all Hamming-distance-1 neighbors — the tightest possible local neighborhood on the hypercube.

After the weighted sum: optional bias, then ReLU activation. Standard CNN mechanics.

## Why it works

**Vertex-transitive symmetry.** Every vertex of a binary hypercube is structurally identical to every other. Weight sharing is not an approximation (as it arguably is at image boundaries in spatial CNNs) — it is exact. The kernel applied at vertex 0 sees the same local geometry as the kernel at vertex 571.

**Composable receptive fields.** One layer sees Hamming distance 1. Two stacked layers see distance 2. After DIM layers, every vertex can influence every other vertex. The hypercube diameter is only DIM (logarithmic in N), so global information propagates fast.

**Bitwise computation.** Neighbor lookup is a single XOR instruction. No pointer chasing, no adjacency storage, no hash tables. The entire topology is implicit in the bit representation of vertex indices.

**Parameter efficiency.** The kernel has K = DIM weights per input/output channel pair. At DIM=10, that's 10 weights — comparable to a 3x3 spatial kernel (9 weights). The architecture is not wider than a standard CNN; it's shaped differently.

## Pooling

Antipodal pooling pairs each vertex v with its bitwise complement `v ^ (2^DIM - 1)`, the maximally distant vertex. One survives (max or average). This reduces DIM by 1 — the output is a perfect (DIM-1)-dimensional hypercube.

The analogy to 2x2 spatial max-pooling holds, but instead of collapsing adjacent pixels, antipodal pooling collapses maximally separated vertices — capturing the widest possible context per reduction step.

## What it's for

HypercubeCNN is designed for data that naturally lives on binary hypercubes — or can be meaningfully mapped onto them:

**Native hypercube data** (the sweet spot):
- **Molecular fingerprints** — ECFP and MACCS keys are binary vectors (1024 or 2048 bits). Each molecule is literally a hypercube vertex. Molecular similarity is measured by Hamming distance (Tanimoto similarity). The convolution kernel operates in exactly the geometry chemists use to compare molecules.
- **Combinatorial feature interactions** — any domain with N binary features where the full binary vector is the input and interactions between features matter for classification.

**Embedded data** (works, but handicapped):
- **Image classification** — pixels are mapped onto hypercube vertices via direct linear assignment. The spatial locality of the original image is not preserved. The network must learn all useful relationships from the hypercube topology alone. MNIST reaches ~98% accuracy — respectable but below spatial CNNs (~99.3%) that get 2D locality for free.

The research direction: native hypercube data is where the architecture has a structural advantage over spatial CNNs. Spatial embeddings (like MNIST) work but sacrifice the locality that grid-based networks get for free.

## Current results

**MNIST** (no spatial inductive bias):
- 60K train / 10K test, 4 conv+pool stages, ~200K parameters
- **98.10%** test accuracy with Xavier/He init, SGD-momentum, cosine LR, L2 weight decay
- NN-only kernel (K=DIM) outperforms the earlier shell-mask design by +1.8% accuracy and 1.58x speedup

See [mnist.md](mnist.md) for the full benchmark.

## Implementation

Pure C++23 with zero external dependencies, distributed as the `HypercubeCNNCore` static library. All public symbols live in `namespace hcnn`. The canonical SDK front door is the `hcnn::HCNN` class — a single PIMPL-style wrapper around the entire pipeline (input embedding → conv/pool stack → readout) with PascalCase methods. Power users who need direct weight access (serialization, gradient checking, custom training loops) can reach the underlying layer classes (`HCNNConv`, `HCNNPool`, `HCNNReadout`, `HCNNNetwork`) through transitive re-exports.

Threading uses a custom fork-join `ThreadPool` (no OpenMP, no GPU, no external runtime). Three threading strategies coexist but never nest: per-sample batch parallelism for training and inference, plus per-vertex parallelism inside conv layers when DIM is large enough to make fork-join overhead worthwhile.

The optimizer is configurable per-network: SGD-with-momentum (default) or Adam with decoupled weight decay. Conv layers support optional batch normalization, ReLU / LeakyReLU / linear activations, and per-channel learnable bias. The readout is either GAP (translation-invariant) or FLATTEN (position-sensitive).

See [architecture.md](architecture.md) for full technical details.
