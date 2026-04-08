# HypercubeCNN Architecture

## Core idea

HypercubeCNN performs convolutions on binary hypercubes instead of spatial grids. The substrate is a DIM-dimensional binary hypercube with N = 2^DIM vertices. All geometry is bitwise — neighbor lookup is a single XOR, there are no adjacency lists, no padding, no border cases.

The key insight: on a binary hypercube, every vertex has exactly DIM nearest neighbors at Hamming distance 1 (single-bit flips). Our convolution kernel learns one weight per neighbor direction (per bit position), shared across all vertices — the hypercube analogue of a 3x3 spatial kernel shared across all pixel positions.

## Data representation

### Vertices

Each vertex is identified by a DIM-bit integer in [0, N). The vertex index IS the coordinate — bit k of the index corresponds to dimension k of the hypercube.

### Channels

One channel = one scalar value per vertex = N floats. Multiple channels are independent copies of the same hypercube geometry, stored contiguously in a channel-major layout:

```
activations[c * N + v]  →  channel c, vertex v
```

Example at DIM=10: one channel = 1024 floats. 64 channels = 65,536 floats in one flat array.

### Memory layout

All data flows through the pipeline as flat `float*` arrays. No tensors, no multidimensional containers. The channel-major convention is universal: conv input, conv output, pool input/output, readout input.

## The convolution (`HCNN`)

### What it computes

For each output channel co, each vertex v:

```
out[co, v] = bias[co] + sum over (ci, k) of w[co, ci, k] * in[ci, v ^ (1 << k)]
```

where:
- `ci` iterates over input channels
- `k` iterates over [0, DIM) — the DIM bit-flip directions
- `v ^ (1 << k)` is the neighbor of v obtained by flipping bit k (XOR)
- `w[co, ci, k]` is the learned weight for output channel co, input channel ci, direction k

This is a direct analogue of a spatial convolution: instead of sliding a 3x3 kernel across a 2D grid, we apply a DIM-direction kernel at every hypercube vertex. The kernel is indexed by bit-flip direction, not by spatial offset.

### Kernel shape

```
kernel[co * c_in * K + ci * K + k]     where K = DIM
```

Total kernel parameters per layer: `c_out * c_in * K`. At DIM=10 with 64→128 channels: 64 * 128 * 10 = 81,920 weights.

For comparison, a standard 3x3 conv with the same channel dimensions: 64 * 128 * 9 = 73,728. Similar ballpark — the hypercube kernel has DIM directions where a spatial kernel has 9 (or 25 for 5x5).

### Why this works

The binary hypercube is vertex-transitive: every vertex looks the same from every other vertex. This means weight sharing is mathematically principled, not just a heuristic — the same kernel applied at every vertex respects the symmetry group (Z_2^n).

Each bit-flip direction k is a distinct geometric axis of the hypercube. The kernel learns how much each axis matters for each input→output channel pair. After multiple layers, information from vertices at Hamming distance 2, 3, ... arrives through composition of single-bit-flip convolutions (like how stacked 3x3 convolutions see larger receptive fields).

### Activation and bias

After the weighted sum, each output vertex optionally gets:
1. A per-channel bias term (one scalar per output channel, added to all vertices)
2. ReLU activation: `out = max(0, pre_activation)`

Both are standard and identical to spatial CNNs.

### Cache tiling

The inner vertex loop is tiled with T=64 vertices per tile. Within a tile:
- Bit-flip masks with `k < log2(T)` stay within the same tile (cache-hot)
- Higher-bit masks map to exactly one other tile (predictable prefetch)

The tile loop structure is: `for tile: for ci: for k: for v_in_tile`, keeping the output tile in L1 for the entire accumulation + activation sequence.

## Pooling (`HCNNPool`)

### Antipodal pooling

Pairs each vertex v with its bitwise complement (antipode) `v ^ (2^DIM - 1)`, the maximally distant vertex on the hypercube. Reduces DIM by 1: the lower-half vertex (bit DIM-1 = 0) survives.

```
For each channel, each vertex v in [0, N/2):
    v_anti = v ^ ((1 << DIM) - 1)
    MAX:  out[v] = max(in[v], in[v_anti])
    AVG:  out[v] = (in[v] + in[v_anti]) / 2
```

Output is a perfect (DIM-1)-dimensional hypercube. The pooled geometry is exact — no approximation, no interpolation.

### Why antipodal?

On a binary hypercube, the antipodal vertex is the most information-rich pairing: it's at maximum Hamming distance, so combining it with the original vertex captures the widest possible context in a single reduction. This is the hypercube analogue of max-pooling over a 2x2 spatial patch, but instead of collapsing adjacent pixels, it collapses maximally distant vertices.

## Readout (`HCNNReadout`)

1. **Global average pooling**: average each channel across all remaining vertices → one scalar per channel.
2. **Linear layer**: `[c_final] → [num_classes]` with bias. Produces raw logits.

This is identical to the GAP + FC readout used in modern spatial CNNs (ResNet, etc.).

## Network assembly (`HCNNNetwork`)

The network is built by stacking conv and pool layers sequentially:

```cpp
HCNNNetwork net(10);              // DIM=10, N=1024
net.add_conv(32);                 // 1→32 channels,   K=10 (DIM=10)
net.add_pool(PoolType::MAX);      // DIM 10→9,        N 1024→512
net.add_conv(64);                 // 32→64 channels,  K=9  (DIM=9)
net.add_pool(PoolType::MAX);      // DIM 9→8,         N 512→256
net.add_conv(128);                // 64→128 channels, K=8  (DIM=8)
net.add_pool(PoolType::MAX);      // DIM 8→7,         N 256→128
net.add_conv(128);                // 128→128 channels, K=7  (DIM=7)
net.add_pool(PoolType::MAX);      // DIM 7→6,         N 128→64
net.randomize_all_weights();      // Xavier/Glorot init
```

Key properties:
- DIM shrinks only at pool layers. Conv layers preserve dimensionality.
- K = DIM at each conv layer, so deeper layers have fewer kernel directions (but operate on more abstract features).
- Channel count typically increases with depth (same as spatial CNNs).
- The readout is automatically configured from the final channel count.

## Input embedding

The embedding maps external data onto hypercube vertices. Currently uses **Direct Linear Assignment**: the first min(input_length, N) scalars are assigned to vertices 0, 1, 2, ... in index order. Remaining vertices are zero-padded.

For MNIST (784 pixels → 1024 vertices): pixels land on vertices 0-783, vertices 784-1023 are zero. The mapping is simple and intentionally structure-agnostic — no spatial locality is encoded. The network must learn all useful relationships from the hypercube topology alone.

## Weight initialization

Xavier/Glorot uniform (default): `scale = sqrt(6 / (fan_in + fan_out))` where `fan_in = c_in * K` and `fan_out = c_out * K`. Computed per-layer since K = DIM varies after pooling.

## Training

### Optimizer

SGD with momentum (0.9) and optional L2 weight decay. Decay is applied to kernel and readout weights only (not biases). The update rule:

```
g = gradient + weight_decay * weight
velocity = momentum * velocity + g
weight -= learning_rate * velocity
```

### Learning rate schedule

Cosine annealing from `lr_max` to `lr_min` (1e-5) over the full training run:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

No warmup, no restarts. Smooth decay avoids the instability of step decay.

### Mini-batch parallelism

`train_batch` processes B samples in parallel across threads. Each thread runs full forward + backward, accumulating gradients into thread-local buffers. After all samples complete, gradients are reduced (summed), averaged, and applied in a single weight update.

Per-layer vertex threading is automatically disabled during batch parallelism to prevent nested reentrancy on the non-reentrant ThreadPool.

## Parameter count

For the current MNIST configuration (DIM=10, 4 stages):

| Layer | Shape | Kernel | Bias | Total |
|-------|-------|--------|------|-------|
| Conv1 | 1→32, K=10 | 320 | 32 | 352 |
| Conv2 | 32→64, K=9 | 18,432 | 64 | 18,496 |
| Conv3 | 64→128, K=8 | 65,536 | 128 | 65,664 |
| Conv4 | 128→128, K=7 | 114,688 | 128 | 114,816 |
| Readout | 128→10 | 1,280 | 10 | 1,290 |
| **Total** | | | | **~200K** |

## Implementation

All core code is in the `HypercubeCNNCore` static library (pure C++23, no external dependencies). The library exports:

| Class | File | Role |
|-------|------|------|
| `HCNN` | HCNN.h/cpp | Single conv layer |
| `HCNNPool` | HCNNPool.h/cpp | Antipodal pooling layer |
| `HCNNReadout` | HCNNReadout.h/cpp | GAP + linear readout |
| `HCNNNetwork` | HCNNNetwork.h/cpp | Pipeline orchestrator |
| `ThreadPool` | ThreadPool.h | Header-only fork-join pool |

Executables are thin wrappers that link the library. This separation is intentional — the library is the future C++ SDK surface.
