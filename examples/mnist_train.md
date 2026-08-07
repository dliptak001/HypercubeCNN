# MNIST Training — Handwritten Digit Classification

End-to-end training and evaluation of HypercubeCNN on MNIST (60K train, 10K test, 10 classes: images of digits 0–9).

**Headline (closed recipe):** ~**99.44%** mean best test accuracy over **3 weight-init seeds** (range **99.42–99.46%**). Not a leaderboard claim — see [Results](#results) and the honesty notes below.

## What this example shows

- Loading MNIST from IDX files
- **Spatial preprocess:** `HCNNSpatialAugmenter` (train) → `HCNNSpatialEmbedder` DualPlaneResize (DIM=11, N=2048)
- **Train-time aug:** rot / scale / shift / **shear_x** (default); elastic **off**; light Gaussian noise
- **`DemoConfig`** at the top of `mnist_train.cpp` — one place for seed, schedule, aug, dim, layers
- Architecture / train: public `HCNNConfig::Build`, `LayerSpec`, `HCNNTrainer`;
  example-only knobs stay in `DemoConfig` (data, aug, embed, logging)
- Adam, cosine LR, weight decay, dual checkpoints (`HCNNDualCheckpoint`)
- Train helpers: `cosine_lr`, `evaluate_classification`, `HCNNFlatDataset`

## How MNIST maps onto the hypercube

### Loader

28×28 = 784 grayscale pixels, normalized to **[-1, 1]** (background ≈ −1).

### Spatial pipeline (always length 2048)

`input_length = N = 2^11 = 2048` for every train/infer call. DualPlaneResize at DIM=11 fills the cube exactly (pattern length 2048).

```text
28×28 digit in [-1, 1]
        │
        │  (train only) HCNNSpatialAugmenter
        │    affine: rot ±12°, scale [0.9, 1.1], shift ±2, shear_x ±0.15
        │    elastic: off
        │    N(0, 0.03²), clip after noise
        ▼
28×28 (possibly warped)
        │
        │  HCNNSpatialEmbedder  DualPlaneResize  pad_value = -1
        ▼
 out[0 .. 1023]      = 32×32 bilinear resize (ink)
 out[1024 .. 2047]   = 32×32 |∇| max-normed to about [-1, 1]
```

| Region | Content |
|--------|---------|
| Vertices 0–1023 | 32×32 ink plane |
| Vertices 1024–2047 | 32×32 gradient magnitude of that plane |

Layout is **row-major blocks**, not Hamming-local packing. Goal: full occupancy + multi-view (ink ‖ edges). See [`docs/spatial_preprocess.md`](../docs/spatial_preprocess.md).

**Pad contract:** after spatial embed, always pass `input_length = N`. Network Embed zero-pads short tails and would wipe a non-zero `pad_value`.

Test packing uses the same embed path with **no** aug (`HCNNSpatialAugConfig::None()`).

## Architecture

All knobs: **`DemoConfig`** in `mnist_train.cpp`.

**Default stack:**

```text
Input: SpatialEmbed DualPlane  2048 floats  (DIM=11, 1 channel)
  |
Conv1: 1  -> 16, K=12 (DIM+1 self+neighbors), RELU, bias
Conv2: 16 -> 16, K=12, RELU, bias
Conv3: 16 -> 16, K=12, RELU, bias
  |
FLATTEN -> Linear(32768 -> 10) -> logits
```

| Component | Params |
|-----------|-------:|
| Conv1 (1×16×12 + 16 bias) | 208 |
| Conv2 (16×16×12 + 16 bias) | 3,088 |
| Conv3 (16×16×12 + 16 bias) | 3,088 |
| Readout (16×2048 → 10 + bias) | 327,690 |
| **Total** | **334,074** |

**No antipodal pool** — DIM stays 11 / N stays 2048 so both DualPlane halves stay addressable into the fat FLATTEN head. Antipodal MAX on this pack pairs ink-half with grad-half addresses and halves the head; skipped for this recipe.

Edit `DemoConfig` for other stacks (pools, width, BN, etc.). BN is available per conv but is **not** part of the documented accuracy recipe.

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | Adam | Decoupled weight decay (AdamW-style); default betas |
| LR | `lr_max = 1e-3` | Cosine to `lr_min = 0.1 × lr_max` (= **1e-4**) |
| Schedule | `cosine_lr` | Progress `epoch / (epochs − 1)`; last epoch hits floor |
| Batch | 256 | Via `TrainEpoch` → `TrainBatch` |
| Weight decay | 1e-3 | Kernels + readout weights (not biases) |
| Epochs | **100** | |
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` (fixed stream; independent of weight seed) |
| Weight init seed | **398479293** (default) | Change `weight_seed` only for multi-seed probes |
| Aug (train only) | rot ±12°; scale [0.9, 1.1]; shift ±2; **shear_x ±0.15**; shear_y **0**; **elastic off**; noise σ=0.03; OOB = −1; rebuilt each epoch | |
| Checkpoints | dual | Best test **loss** and best test **acc**; process leaves net on **best-acc** weights |

## Data loading

IDX load: `load_mnist()` in `dataloader/HCNNDataset.h` (784-vectors). Aug + embed + flat buffers are built in `mnist_train.cpp`.

`MNISTTrain` does **not** read MNIST from this git clone. It loads **only** from
the local deploy folder `C:\HypercubeCNN\data` (`examples/find_data_dir.h`).

```cpp
const auto data_dir = hcnn_ex::FindMnistDataDir(argv0);
auto train_raw = load_mnist((data_dir / "train-images-idx3-ubyte").string(),
                            (data_dir / "train-labels-idx1-ubyte").string(), 60000);
auto test_raw  = load_mnist((data_dir / "t10k-images-idx3-ubyte").string(),
                            (data_dir / "t10k-labels-idx1-ubyte").string(),  10000);
// fill_spatial_dataset: optional SpatialAug -> SpatialEmbed DualPlane -> FlatDataset
```

### MNIST files (not in git)

**Location (required):** `C:\HypercubeCNN\data\`

**Required names** (uncompressed IDX):

```text
train-images-idx3-ubyte    (~45 MB)
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Download once into that folder (example):

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

On Windows, any tool that fetches and decompresses those four files into
`C:\HypercubeCNN\data` is fine.

MNIST: Yann LeCun, Corinna Cortes, Christopher J.C. Burges. This repo ships only the loader.

## How to run

```bash
cmake --build cmake-build-release --target MNISTTrain
# MinGW runtime DLLs must be on PATH when running the exe (Windows)
./cmake-build-release/MNISTTrain
```

## Results

**Recipe (fixed for all seeds below):** 60K/10K, DIM=11, 3× Conv16 RELU, no pool, DualPlane 32×32 ‖ |∇|, train aug as in the table above (elastic **off**), Adam, batch 256, wd 1e-3, cosine 1e-3 → 1e-4, **100 epochs**, K = DIM+1. Only **weight init seed** changes; aug/shuffle streams fixed. Wall clock includes pack+aug rebuild + `TrainEpoch` (~1.0–1.1k samples/s on 32 threads, ~56–59 s/epoch). Params: **334,074**.

**How to quote:** prefer **3-seed mean best-acc ~99.44%**. A single peak needs the printed `Weight init seed`. This is a **demo pack** (engineered embed + aug + fat FLATTEN), not a claim of free 2D CNN inductive bias or SOTA MNIST.

### Multi-seed (weight init only)

| Weight seed | Best acc | Loss @ best-acc | Best loss | Acc @ best-loss |
|-------------|----------|-----------------|-----------|-----------------|
| **398479293** (default) | **99.46%** | 0.01646 @ ep **60** | **0.01554** @ ep **81** | **99.45%** |
| **287821292** | **99.43%** | 0.01937 @ ep **98** | **0.01746** @ ep **71** | **99.42%** |
| **498279213** | **99.42%** | 0.01818 @ ep **81** | **0.01814** @ ep **97** | **99.39%** |

| Statistic | Value (3 seeds) |
|-----------|-----------------|
| **Mean best-acc** | **99.437%** (~**99.44%**) |
| Range (best-acc) | **0.04 pp** (99.42–99.46) |
| Mean best-loss CE | **0.01705** |
| Mean acc @ best-loss | **99.420%** |

All three seeds clear **99.4%** best-acc. Init variance on peak accuracy is small. Best-loss CE is slightly looser (0.0155–0.0181) but acc@best-loss stays ~99.4%.

### Checkpoints

**Seed 398479293 (default)**

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| Best loss | 81 | 0.01554 | 99.45% (9945/10000) |
| Best acc | 60 | 0.01646 | **99.46%** (9946/10000) |

First ≥99%: epoch **16** (99.14%).

**Seed 287821292**

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| Best loss | 71 | 0.01746 | 99.42% (9942/10000) |
| Best acc | 98 | 0.01937 | **99.43%** (9943/10000) |

First ≥99%: epoch **19** (99.00%).

**Seed 498279213**

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| Best loss | 97 | 0.01814 | 99.39% (9939/10000) |
| Best acc | 81 | 0.01818 | **99.42%** (9942/10000) |

First ≥99%: epoch **20** (99.04%). Dual checkpoints nearly coincide on CE (~0.0182).

### Curves (selected epochs)

**398479293**

```text
Epoch  Acc     Loss     note
  1    96.92%  0.1155
 16    99.14%  0.0270   first ≥99%
 60    99.46%  0.0165   best acc
 81    99.45%  0.0155   best loss
100    99.31%  0.0177
```

**287821292**

```text
Epoch  Acc     Loss     note
  1    97.49%  0.0941
 19    99.00%  0.0301   first ≥99%
 71    99.42%  0.0175   best loss
 98    99.43%  0.0194   best acc
100    99.42%  0.0191
```

**498279213**

```text
Epoch  Acc     Loss     note
  1    96.51%  0.1208
 20    99.04%  0.0292   first ≥99%
 81    99.42%  0.0182   best acc
 97    99.39%  0.0181   best loss
100    99.35%  0.0197
```

## Design notes

### Dense DualPlane pack

Zero-padding most of N wastes capacity at DIM=11. Ink ‖ |∇| fills every vertex with a simple multi-view; it does **not** make bit flips equal 2D adjacency.

### No antipodal pool

Correct math, poor fit for this layout (cross-half ink/grad pairs) and it cuts FLATTEN capacity. Full N=2048 is intentional.

### FLATTEN head only

`HCNNReadout` is linear over `num_features = c × N`. GAP over vertices underperformed early and is not in the SDK head. The head stays position-addressable.

### Self-tap kernels (K = DIM + 1)

Center + Hamming-1 directions. With this recipe, 3-seed best-acc lands in **99.42–99.46%** (~**99.44%** mean), above earlier neighbor-only / shorter-schedule demos (~99.3%). Cost is ~1/DIM extra params per conv.

### Augmentation

FLATTEN is position-sensitive; affine (including **shear_x**) reduces absolute vertex memorization. **Elastic (α=1, σ=5)** was tried on the default seed: slightly **worse** peak acc and CE, slower epochs — leave **off**. Aug is train-only; test metrics use clean IDX digits.

### Curve shape

~97% after epoch 1; first ≥99% by epoch **16–20**; then a long climb. Dual-checkpoint timing varies by seed (mid-run vs late). Best-acc range across seeds is only **0.04 pp**.

### Context for ~99.44%

| Baseline (rough) | Acc |
|------------------|-----|
| Linear on pixels | ~92% |
| Small MLP | ~98% |
| Typical spatial CNNs | 99.0–99.5%+ |

HypercubeCNN at **~99.44% mean best-acc** reaches **solid spatial-CNN accuracy** while learning on **hypercube geometry** (DualPlane pack + self/neighbor Hamming kernels + FLATTEN), not a 2D convolution grid. Remaining errors ~54–58/10k at best-acc.

## Significance

The closed 3-seed survey (**99.46% / 99.43% / 99.42%** best-acc) shows the demo stack is solid: **self-tap kernels + depth + full-N FLATTEN + affine train aug** on an engineered image embedding. Pack, aug, schedule, and seeds are **image-demo engineering**, separate from native cube workloads (ESN state, fingerprints, Boolean functions). For API usage see [`docs/CPP_SDK.md`](../docs/CPP_SDK.md); for packing contracts see [`docs/spatial_preprocess.md`](../docs/spatial_preprocess.md).
