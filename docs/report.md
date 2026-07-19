# HypercubeCNN — Core Technical Report

**Source of truth:** this report is grounded in the **code modules** under the project root (`HCNN*.h/cpp`, `ThreadPool.h`, optional spatial/train helpers, examples, and tests). The top-level `README.md` and some narrative docs lag the implementation; where they conflict, **the code wins**.

| | |
|---|---|
| **Library** | `HypercubeCNNCore` (C++23, CMake ≥ 3.21, Apache-2.0) |
| **Version (CMake)** | 0.2.0 |
| **Public front door** | `hcnn::HCNN` (`HCNN.h`) |
| **Namespace** | `hcnn` |

---

## 1. Concept and implications

### 1.1 What it is

HypercubeCNN is a **convolutional neural network whose spatial domain is a Boolean hypercube graph**, not a 2D pixel grid.

- Topology: a `DIM`-dimensional binary hypercube with **N = 2^DIM** vertices, addressed by integer indices in `[0, N)`.
- Connectivity: vertex `v` has exactly **DIM** nearest neighbors at Hamming distance 1, obtained by flipping one bit: `v ^ (1 << k)` for `k ∈ [0, DIM)`.
- Values: activations are ordinary **float** scalars (commonly in `[-1, 1]`). The hypercube is the **graph data lives on**, not a constraint that values be bits.
- Learning: kernels and readout are trained end-to-end with backpropagation (SGD+momentum or Adam/AdamW-style updates).

The same XOR-addressed topology family is shared with sibling projects (HypercubeRC, HypercubeHopfield). Here the emphasis is **learned multi-channel convolution + hierarchical optional pooling + linear readout**.

### 1.2 Core convolution idea

For each output channel `co` and vertex `v`:

```text
out[co, v] = bias[co]
           + sum over ci, k of  w[co, ci, k] * in[ci, v ^ (1 << k)]
```

with:

| Symbol | Meaning in code |
|--------|-----------------|
| `ci` | input channel |
| `k` | bit-flip direction; **K = DIM** |
| `w[co, ci, k]` | weight shared across **all** vertices |
| layout | channel-major: `data[c * N + v]` |

This is the hypercube analogue of a shared spatial kernel: one weight per **axis of the cube**, not per absolute address. There are **no adjacency lists, no padding, and no border special cases** — neighbor lookup is a single XOR.

**Kernel taps:** each layer has **K = DIM + 1** weights per (input channel, output channel): one **self/center** tap multiplying `in[ci, v]` plus **DIM** Hamming-1 neighbor taps (bit flips). This is the hypercube analogue of a spatial kernel that includes the center pixel (degree-1 graph filter `a0 I + A` on the cube adjacency).

### 1.3 Why the topology matters (implications)

1. **Exact weight sharing.** The hypercube is vertex-transitive. Shared weights are symmetry-correct for the graph, not an implementation convenience that breaks at image borders.

2. **Geometry is defined by packing.** Bit `k` is a **global involution** of the address space. What “direction k means” for an application depends entirely on how external data is mapped onto vertex indices. Row-major image packing does **not** make Hamming-1 equal to 4-neighborhood on a 2D grid.

3. **Depth = multi-hop mixing.** One conv layer mixes Hamming distance 1. Stacked convs compose; after L layers, information can travel roughly Hamming distance L (and more via multi-path mixing), analogous to receptive-field growth in spatial CNNs.

4. **Capacity grows exponentially with DIM.** N doubles per +1 DIM. Memory and compute for dense activations scale as `channels × N`. Practical training demos live around DIM 6–12; the hard limit in code is `3 ≤ DIM ≤ 32`.

5. **Pooling is antipodal, not local.** Pooling pairs `v` with `v ^ (2^DIM - 1)` (maximum Hamming distance) and produces a perfect (DIM−1)-subcube of size N/2. That is a strong, non-spatial reduction — very different from 2×2 max-pool on an image.

6. **Readout is position-addressable by default.** The network rebuilds the readout as a linear map over **every** final `(channel, vertex)` feature (see §4.5). Vertex identity is preserved into the head. Global average pooling over vertices is **not** the default path used by the orchestrator.

7. **Not a free graph CNN.** The edge set is fixed by the cube; you cannot attach arbitrary neighborhoods without changing the primitive.

### 1.4 What the project is *not*

- Not a drop-in replacement for ImageNet-scale 2D CNNs / ViTs.
- Not a bit-vector network or Boolean circuit learner out of the box (values are float).
- Not dependent on spatial preprocess for native hypercube workloads (spatial modules are optional).
- Not GPU-native today (CPU static library, fork-join `ThreadPool`, SIMD-friendly block-pair loops on the non-threaded path).

---

## 2. Potential applications

### 2.1 Where it may excel

| Domain | Why the fit is natural |
|--------|-------------------------|
| **Reservoir / HypercubeRC readout** | State already lives on a length-N cube; conv shares structure across vertices while FLATTEN preserves unit identity (timescales, modes). Regression + `TANH` path is first-class. |
| **Native hypercube / bit-index features** | Boolean functions, binary fingerprints, XOR-structured codes, Gray-code / product-space features where Hamming locality is meaningful. |
| **Structured vector fields of length 2^D** | Any feature vector whose coordinates are naturally labeled by a D-bit address and where shared bit-axis filters make sense. |
| **Small / mid N with strong packing** | When you control the map into vertices (or data already is cube-shaped), and N is thousands–tens of thousands, CPU training is practical. |
| **Position-sensitive regression / classification** | Tasks where **which vertex** fired matters more than translation invariance. FLATTEN + linear head exploits that. |
| **Embedded / dependency-light C++** | Pure stdlib + threads; FetchContent / `find_package` SDK; no autodiff framework required. |
| **Teaching / research on alternate geometries** | Clean laboratory for “what if CNN geometry is Z₂ⁿ instead of Z²?” with real backprop and optimizers. |

**Evidence from in-tree demos (teaching, not leaderboard claims):**

- **MNIST** (`examples/mnist_train.cpp`): engineered 2D→cube pack (DualPlane), no pool default, Adam + cosine LR; code comments document ~**99.28%** best-acc on a named seed under the current recipe. This shows the *training stack* works and that a fat FLATTEN head + aug can classify digits without a 2D grid prior — not that hypercube geometry is optimal for vision.
- **Regression timeseries** (`examples/regression_timeseries.cpp`): synthetic uncoupled reservoir → next sine sample; documents near-perfect R² as an **API/smoke** signal, explicitly not proof of live HypercubeRC skill.

### 2.2 Applications to avoid (or treat with caution)

| Domain / pattern | Reason |
|------------------|--------|
| **Large natural images / detection / segmentation** | Need huge N or aggressive downsampling; no pyramid of spatial strides, no multi-scale spatial priors, no GPU stack. Use a standard vision CNN/transformer. |
| **Tasks that require translation equivariance on a 2D grid** | Hamming kernels do not implement translation unless packing + architecture invent it. Row-major DualPlane packing is **not** locality-preserving. |
| **Relying on antipodal pool as “spatial max-pool”** | Antipodes are **maximally distant** addresses. On DualPlane packs, every pair can straddle ink-half vs grad-half; demos often **skip pool** for that reason. |
| **GAP-style hierarchy as the main story** | Early MNIST experiments (documented in example notes) found GAP over vertices weak when packing is not locality-aware and features are not translation-tolerant. Code path is FLATTEN. |
| **RGB / multi-view without custom packing** | Spatial helpers are **single-channel**. `input_channels > 1` is supported by the core, but you own layout. |
| **“HypercubeCNN will beat spatial CNNs on MNIST/CIFAR”** | Competitive numbers on MNIST come with heavy **demo engineering** (pack, aug, fat head). Spatial CNNs remain the default for grid data. |
| **Very high DIM training without memory plan** | N = 2^DIM; activations and FLATTEN weights scale accordingly. |
| **Production training resumption from weight blobs alone when BN is on** | `GetWeights` / `SetWeights` omit BN γ/β and all optimizer state (see §6). |
| **Chaotic multi-step forecasting claims from the sine demo** | The demo’s reservoir is uncoupled; near-zero MSE does not transfer by assertion. |

### 2.3 Design sweet spot (summary)

HypercubeCNN is strongest when:

1. Data **already** (or naturally) lives on a hypercube / power-of-two address space, **or**
2. You deliberately design a packing so that **bit axes carry meaning**, **and**
3. You want a **small/medium** learned conv stack + **position-aware** linear head in pure C++.

It is weakest when you mainly need classical 2D locality, large vision scale, or framework ecosystem features (GPU, ONNX zoo, automatic mixed precision, etc.).

---

## 3. Architecture (modules and pipeline)

### 3.1 Pipeline

```text
raw floats (caller)
    │
    ├─ optional: HCNNSpatialAugmenter   (2D only; DIM-agnostic)
    ├─ optional: HCNNSpatialEmbedder    (2D → length N = 2^dim)
    │
    ▼
HCNN::Embed  (Direct Linear Assignment + zero pad to capacity)
    │
    ▼
[ HCNNConv  →  optional HCNNPool ]*   (DIM constant on conv; −1 on pool)
    │
    ▼
HCNNReadout  (FLATTEN linear: c_final * N_final → num_outputs)
    │
    ▼
raw logits (classification) or raw predictions (regression)
```

Task type does **not** change the forward stack — only loss gradient and training API shape.

### 3.2 Module map (code)

| Module | Files | Role |
|--------|-------|------|
| **HCNN** | `HCNN.h/cpp` | SDK front door: architecture builder, train/infer epoch helpers, shuffle gather, weight blob I/O. Owns `unique_ptr<HCNNNetwork>`. Non-copyable, non-movable. |
| **HCNNNetwork** | `HCNNNetwork.h/cpp` | Orchestrator: layer list, embed, forward, shared train cores, BN/thread RAII guards, buffer pools. |
| **HCNNConv** | `HCNNConv.h/cpp` | Hypercube conv: XOR-neighbor accumulate, bias, optional BN, activation, SGD/Adam, dual backward APIs. |
| **HCNNPool** | `HCNNPool.h/cpp` | Antipodal MAX/AVG; no parameters. |
| **HCNNReadout** | `HCNNReadout.h/cpp` | Linear head with optional per-channel average over N; network always uses FLATTEN mode (N=1, features = c×N_final). |
| **ThreadPool** | `ThreadPool.h` | Header-only fork-join pool; caller is thread 0; **not reentrant**. |
| **HCNNSpatialAug** | `HCNNSpatialAug.h/cpp` | Optional 2D affine / elastic / noise. |
| **HCNNSpatialEmbed** | `HCNNSpatialEmbed.h/cpp` | Optional 2D→N pack (RowMajorPad, ResizeToFit, DualPlaneResize). |
| **HCNNTrainHelpers** | `HCNNTrainHelpers.h/cpp` | Optional metrics, cosine LR, checkpoints, flat classification dataset. |
| **Examples / tests** | `examples/`, `tests/`, `dataloader/` | Teaching demos and smoke tests — not part of the mathematical core. |

CMake target: **`HypercubeCNNCore`** static library. Options of note: `HCNN_FAST_TANH` (default ON), `HCNN_NATIVE_ARCH`, `HCNN_FAST_MATH`, `HCNN_BUILD_EXAMPLES`.

### 3.3 Incremental network build

```cpp
hcnn::HCNN net(start_dim, num_outputs, input_channels,
               hcnn::TaskType::Classification,  // or Regression
               hcnn::LossType::Default,
               /*num_threads=*/0);

net.AddConv(c_out, hcnn::Activation::RELU, /*bias=*/true, /*bn=*/false);
net.AddPool(hcnn::PoolType::MAX);   // optional; DIM -= 1
// ... more stages ...
net.RandomizeWeights();             // sizes readout; He/Xavier per layer
net.SetOptimizer(hcnn::OptimizerType::ADAM);
```

Constraints enforced in code:

- `3 ≤ start_dim ≤ 32`
- `input_channels ≥ 1`, `num_outputs ≥ 1`
- Loss/task pairing: Classification ↔ CrossEntropy only; Regression ↔ MSE only (`Default` resolves at construction)

### 3.4 Task and loss axes

| Enum | Values | Effect |
|------|--------|--------|
| `TaskType` | Classification, Regression | Training method set; meaning of outputs |
| `LossType` | Default, CrossEntropy, MSE | Gradient of loss w.r.t. outputs |

Shared cores: `train_step_impl` / `train_batch_impl` inject a loss-gradient lambda; forward/backward of layers are task-agnostic.

### 3.5 Optional periphery (not in the graph)

- **Spatial:** aug at native H×W, then embed to N; pass **`input_length = N`** into train/infer if pad is nonzero (network Embed zero-pads tails).
- **Train helpers:** `evaluate_classification`, `evaluate_regression`, `cosine_lr`, `HCNNDualCheckpoint`, `HCNNBestMetricCheckpoint`, `HCNNFlatDataset`.

---

## 4. Mechanics (how the pieces actually work)

### 4.1 Embedding (`HCNNNetwork::embed_input`)

```text
capacity = input_channels * 2^start_dim
for i in [0, input_length):  out[i] = raw[i]
for i in [input_length, capacity): out[i] = 0
```

- Throws if `input_length > capacity`.
- **No runtime clamp** to `[-1, 1]` despite API comments recommending that range.
- “Direct Linear Assignment”: index order, structure-agnostic.

If you already produced a full length-N buffer via spatial embed (including non-zero pad), pass that buffer with `input_length = N` so Embed does not rewrite the tail to zeros.

### 4.2 Convolution (`HCNNConv`)

**Parameters per layer**

| Tensor | Shape |
|--------|-------|
| kernel | `c_out × c_in × K`, `K = DIM + 1` (neighbors `0..DIM-1` + self at `DIM`) |
| bias (optional) | `c_out` |
| BN γ, β (optional) | `c_out` each |
| BN running mean/var | `c_out` each (inference) |

**Forward order**

1. Self + neighbor accumulate (+ bias)
2. Optional batch norm (per-channel stats over **vertices of the current sample**, not over a mini-batch of images)
3. Activation: `NONE`, `RELU`, `LEAKY_RELU` (α=0.01), `TANH`

**BN notes**

- Training: sample stats; EMA of running mean/var (momentum 0.1, eps 1e-5) unless suppressed by `BNStatsGuard` during batch-parallel training (then stats reduced once after the batch).
- Eval: running stats. `EvalModeGuard` forces eval during `forward` / `forward_batch` so inference does not dirty training BN flags/state.

**Init (`randomize_weights`)**

- `scale > 0`: uniform `[-scale, scale]`
- else: He for ReLU/LeakyReLU with `c_in > 1`; Xavier otherwise (including first layer with `c_in = 1` and TANH/NONE)

**Optimizers**

- SGD: velocity + optional L2 on kernels (weight decay); bias typically without the same decay path as kernels in the SGD branch details.
- Adam: bias-corrected m/v; **decoupled weight decay** term on kernels (`lr * (m_hat/sqrt(v_hat) + wd * w)`).
- Global timestep lives on `HCNNNetwork` and is passed into apply paths.

**Performance paths**

- Single-threaded hot path (DIM < 12): **block-pair** rewrite of XOR loads for contiguous SIMD-friendly access.
- Threaded path (DIM ≥ 12): tile vertices (T=64), XOR-indexed; disabled under batch parallelism via `LayerThreadGuard`.
- TANH: optional `HCNN_FAST_TANH` Padé approx; backward can use `1 - y²` from post-activation when provided.

### 4.3 Pooling (`HCNNPool`)

```text
anti_mask = (1 << DIM) - 1
for each channel, for v in [0, N/2):
  MAX: out[v] = max(in[v], in[v ^ anti_mask]); store argmax index
  AVG: out[v] = 0.5 * (in[v] + in[v ^ anti_mask])
```

- Output is a perfect sub-hypercube of dimension DIM−1.
- Channel count unchanged.
- Threading threshold higher than conv (DIM ≥ 14).
- MAX backward requires `max_indices` from forward.

### 4.4 Readout (`HCNNReadout`) — FLATTEN by orchestration

Class implementation:

1. For each of `input_channels` “channels,” average over `N` vertices.
2. Linear map: `num_outputs × input_channels` weights + bias.

Orchestrator always:

```text
readout = HCNNReadout(num_outputs, final_channels * final_N)
// call with N = 1 so average is a no-op
readout.forward(activations, logits, /*N=*/1, ...)
```

So every final `(channel, vertex)` is an independent feature. The class *can* do GAP (N > 1 with `input_channels = c_final`), but **HCNNNetwork never wires that path** after `randomize_all_weights`.

Implication: **parameter count is dominated by the head** whenever final N and channels are large and pools are few/absent — exactly the MNIST demo pattern.

### 4.5 Loss gradients

**Classification / CrossEntropy**

```text
p = softmax(logits)
dL/d logits[i] = class_weight * (p[i] - 1[i == target])
```

Softmax is training-time only; inference returns raw logits.

**Regression / MSE**

```text
dL/d pred[i] = pred[i] - target[i]
```

Comment in code: scale matches a sum-style single-sample convention; callers absorb constants into LR. Metrics helper `evaluate_regression` reports mean MSE and R² separately.

### 4.6 Training dispatch

| API | Behavior |
|-----|----------|
| `TrainStep` / `TrainStepRegression` | Single sample: forward + backward with in-layer weight update |
| `TrainBatch*` | Parallel per-sample forward/grad → reduce mean → one `apply_gradients` |
| `TrainEpoch*` | Chunk into batches; optional deterministic shuffle (`shuffle_seed ≠ 0`) with gather buffers |

Learning rate is **always caller-supplied** (no schedule owned by `HCNN`). Helpers provide `cosine_lr` only.

### 4.7 Threading model

Three strategies, **never nested** on the same pool:

| Strategy | When |
|----------|------|
| Batch sample parallelism | `TrainBatch` / `ForwardBatch`, batch size > 1, pool present |
| Vertex parallelism (conv) | DIM ≥ 12 and not inside batch dispatch |
| Channel parallelism (pool) | DIM ≥ 14 and not inside batch dispatch |

`num_threads = 1` constructs **no** background pool (host-level outer parallelization).

### 4.8 Weight blob

`GetWeights` / `SetWeights` / `GetWeightCount` include:

```text
for each conv: kernel, bias (if enabled)
readout: weights, bias
```

**Not included:** BN γ/β, BN running stats, SGD/Adam moments, Adam timestep. Checkpoints built on the blob are exact for **eval export** on no-BN nets; resume-training and BN nets need extra state.

### 4.9 Spatial preprocess (optional)

| Piece | Knows DIM? | Modes / ops |
|-------|------------|-------------|
| `HCNNSpatialAugmenter` | No | Affine (scale/shear/rotate/shift), optional elastic, noise |
| `HCNNSpatialEmbedder` | Yes | RowMajorPad, ResizeToFit, DualPlaneResize (ink ‖ \|grad\|) |

DualPlane at DIM=11: S=32, 2·S² = 2048 = N (full occupancy) — the MNIST demo default packing. Layout is **row-major blocks**, not Hamming locality-aware maps (those remain a design memo only).

---

## 5. Empirical stacks (what the demos actually build)

These are **code defaults**, not aspirational architecture diagrams.

### 5.1 MNIST (`examples/mnist_train.cpp`)

| Knob | Default in code |
|------|-----------------|
| DIM | 11 (N=2048) |
| Embed | DualPlaneResize, pad = −1 |
| Layers | Conv16 **NONE** → Conv16 **TANH** → Conv16 **RELU** |
| Pool | **none** |
| BN | off |
| Optimizer | Adam, wd=1e-3, batch 256, 60 epochs, cosine 1e-3 → 1e-4 |
| Aug | rot/scale/shift/shear_x; elastic off |

Comment-documented result (single seed `398479293`): **~99.28% best-acc**, **~99.25% at best-loss**. Multi-seed mean is not fully filled in example docs.

**Why no pool in this recipe:** DualPlane halves the address space into ink vs grad; antipodal pairing at DIM=11 is systematically cross-half and also halves FLATTEN capacity. The head is intentionally large (`16 × 2048 → 10`).

### 5.2 Regression timeseries (`examples/regression_timeseries.cpp`)

| Knob | Default in code |
|------|-----------------|
| DIM | 10 (N=1024) |
| Layers | Conv16 RELU → Conv16 TANH |
| Pool | none |
| Task | Regression / MSE |
| Data | Uncoupled leaky tanh units driven by sine |

Documents best test MSE ~3e-8 range as smoke signal; **does not** claim HypercubeRC production accuracy.

---

## 6. Points of confusion (read carefully)

### 6.1 “Boolean hypercube” ≠ Boolean activations

Vertices are bit-indexed; values are floats. Do not assume binarized activations or logical gates.

### 6.2 Kernel width is DIM + 1 (not DIM)

`K = DIM + 1`: indices `0 .. DIM-1` are neighbor bit flips; index `DIM` is the **self/center** tap. Older docs/tags (`pre_self_contribution`) described neighbor-only kernels — that is historical.

### 6.3 Hamming neighbors ≠ image neighbors

Unless packing is designed for it (Gray code / product embedding — not implemented in core), flipping bit `k` is **not** “one pixel to the right.” The MNIST pack optimizes occupancy + multi-view, not spatial↔Hamming alignment.

### 6.4 Antipodal pool is not 2×2 pool

It collapses **maximally distant** pairs and drops DIM by 1. Use it when that reduction is meaningful for the address structure; do not assume it behaves like CNN max-pool on local patches.

### 6.5 FLATTEN vs the readout class name/behavior

`HCNNReadout` implements average-then-linear. The network forces FLATTEN by setting features = `c * N` and `N=1`. Reading only the readout class without the orchestrator suggests GAP; production path is FLATTEN.

### 6.6 Embed pad vs spatial pad

Network Embed **always zero-pads**. Spatial embed may use `pad_value = -1`. Passing a short buffer into Embed after DualPlane/Resize can **wipe** non-zero pad. Contract: train/infer with `input_length = capacity()` after spatial embed.

### 6.7 Docs drift (explicit)

Examples of where narrative lags code:

| Topic | Stale narrative risk | Code reality |
|-------|----------------------|--------------|
| MNIST accuracy in README | Often cites older ~98% | Demo comments ~99.3% class with dense DualPlane + modern recipe |
| MNIST activations | “three RELU” in some notes | **NONE → TANH → RELU** in `mnist_train.cpp` |
| Parameter tables | Deep pool stacks / K=DIM | Current demos often **no pool**, fat FLATTEN; **K=DIM+1** (self+neighbors) |
| Architecture.md examples | Heavy conv+pool ladders | Valid API usage, not the default demo |
| Weight blobs pre-self | Old checkpoints | **Incompatible** with post-self layouts (`pre_self_contribution` tag) |

Trust headers, `.cpp` demos, and this report over marketing snippets.

### 6.8 Weight serialization gaps

BN parameters and optimizer state are live in layers but **absent** from `GetWeights`. Dual checkpoints inherit that gap. For BN training continuity, serialize extra state yourself or re-init optimizer via `SetOptimizer` after restore.

### 6.9 Input range is convention, not enforced

API text says `[-1, 1]`; implementation copies floats as-is. Out-of-range inputs will train but may fight TANH/ReLU assumptions and init scales.

### 6.10 MSE gradient scale

Regression grad is `pred - target` (no `/num_outputs` factor in the backward API). Learning rates are not portable from PyTorch `reduction='mean'` without adjustment.

### 6.11 Thread pool non-reentrancy

Nested `ForEach` deadlocks. Batch training intentionally nulls layer pools. Do not hand-roll nested parallel calls on the same `ThreadPool`.

### 6.12 Single-sample `forward` scratch

`HCNNNetwork::forward` uses shared ping-pong buffers; not safe concurrent with another forward/train on the **same** instance. Use `ForwardBatch` or separate nets for concurrent single-sample work.

### 6.13 `RandomizeWeights` rebuilds readout

Architecture changes that alter final `c` or `N` require re-randomize; readout is reconstructed from final geometry inside `randomize_all_weights`.

### 6.14 Multi-channel input

Supported at the network level (`input_channels`), but spatial helpers and demos are single-channel. Multi-channel packing is entirely the caller’s problem.

### 6.15 Fast tanh and float modes

`HCNN_FAST_TANH` and relaxed math flags trade tiny numeric error for speed. Bit-identical training vs `std::tanh` is not guaranteed when fast tanh is on; demos report still-excellent R² on the sine task.

---

## 7. Strengths and limitations (engineering view)

### Strengths

- Clear mathematical primitive (XOR neighbors + shared directional weights).
- Clean C++ SDK surface (`HCNN`) with contiguous batch APIs.
- Shared train core for classification/regression; extensible loss dispatch.
- Careful threading/BN RAII for parallel mini-batches.
- Allocation-light steady state after buffer warm-up.
- Optional image and train-loop helpers without polluting the core graph.
- Sibling ecosystem fit (HypercubeRC readout consumer).

### Limitations

- CPU-only; no CUDA/Metal path in-tree.
- Exponential N in DIM; FLATTEN heads explode without pooling.
- Spatial inductive bias is **not** free — packing is the missing link for vision.
- Antipodal pooling is specialized; often wrong for current image packs.
- Weight I/O incomplete for BN + optimizer resume.
- Loss zoo is minimal (CE, MSE).
- No center kernel weight; no dilated/strided hypercube ops beyond antipodal pool.
- Documentation surface partially stale relative to demos.

---

## 8. Practical guidance

### Prefer HypercubeCNN when

- Inputs are already length `2^D` (or easily padded) with meaningful addresses.
- You need a learned multi-channel mixing layer on that topology in C++.
- Vertex identity should survive into a linear head (RC units, fingerprints, modes).
- N is modest (roughly ≤ 4k–16k for dense multi-channel stacks on desktop CPU, workload-dependent).

### Prefer something else when

- Primary structure is 2D/3D Euclidean locality at scale.
- You need mature vision tooling, GPU training, or pretrained backbones.
- You require GAP hierarchical features without designing a locality-aware embed.

### Configuration heuristics (from code + demos)

1. **Native cube data:** start DIM to match state length; 1–2 convs; consider TANH near RC-like ranges; pool only if antipodal reduction is meaningful.
2. **Images:** treat spatial aug+embed as mandatory engineering; default DualPlane + **no pool** until packing is locality-aware; watch FLATTEN size.
3. **Regression:** center targets; prefer Adam; match activations to output scale.
4. **Always:** `PrepareBuffers()` if you care about first-batch latency; pass consistent `input_length`; re-`RandomizeWeights` after architecture edits.

---

## 9. File quick reference

```text
HCNN.h / HCNN.cpp              Public pipeline API
HCNNNetwork.h / .cpp           Orchestrator, losses, batching
HCNNConv.h / .cpp              Hypercube convolution + BN + optim
HCNNPool.h / .cpp              Antipodal pool
HCNNReadout.h / .cpp           Linear head (GAP primitive; FLATTEN wired)
ThreadPool.h                   Fork-join pool
HCNNSpatialAug.*               Optional 2D augmentation
HCNNSpatialEmbed.*             Optional 2D → N embed
HCNNTrainHelpers.*             Metrics, LR, checkpoints
examples/mnist_train.cpp       Classification demo (DualPlane recipe)
examples/regression_timeseries.cpp  Regression demo
tests/CoreSmokeTest.cpp        SDK smoke coverage
```

---

## 10. Closing synthesis

HypercubeCNN reimplements the **CNN idea** on the **Boolean hypercube**: shared directional filters along bit axes, optional exact antipodal downsampling, and a dense position-aware readout, all with ordinary float activations and standard first-order optimizers.

Its implications are sharp: **geometry is address algebra**; packing *is* architecture; weight sharing is symmetry-true; capacity and compute track `2^DIM`; and the sweet spot is **structured power-of-two domains and RC-style readouts**, not generic large-scale vision.

Used with eyes open — especially regarding packing, pooling, FLATTEN size, and documentation lag — it is a coherent, production-minded C++ core for an unusual but well-defined computational geometry of learning.
