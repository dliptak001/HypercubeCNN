# MNIST Training -- Handwritten Digit Classification

Demonstrates end-to-end training and evaluation of a HypercubeCNN on the MNIST handwritten digit dataset (60K train, 10K test, 10 classes).

## What this example shows

- Loading real MNIST data from IDX binary files
- **Core spatial preprocess**: `HCNNSpatialAugmenter` (train) → `HCNNSpatialEmbedder` DualPlaneResize (DIM=11, full N=2048)
- **Train-time augmentation**: rot/scale/shift, **shear_x** (default), optional mild elastic (off by default), light Gaussian noise
- **`DemoConfig` at the top of `mnist_train.cpp`**: weight seed, schedule, aug, dim, layer list, sample caps — one place to edit; architecture print + param counts follow
- Mini-batch Adam, cosine LR annealing, weight decay
- Dual checkpoints: best test loss and best test accuracy (`GetWeights` / `SetWeights`)
- Parallel batch inference for evaluation
- Core train helpers (`HCNNTrainHelpers.h`): `cosine_lr`, `evaluate_classification`, `HCNNDualCheckpoint`, `HCNNFlatDataset`

## How MNIST maps onto the hypercube

### Loader

MNIST images are 28×28 = 784 grayscale pixels, normalized to **[-1.0, 1.0]** (background ≈ −1).

### Spatial pipeline (always length 2048)

Before `TrainEpoch` / `ForwardBatch`, each image goes through the core helpers. Embed always writes a **full** length-N buffer (`input_length = N = 2^11 = 2048`). At DIM=11 DualPlaneResize with auto side, pattern length is exactly 2048 (no pad tail).

```
28×28 digit in [-1, 1]
        │
        │  (train only) HCNNSpatialAugmenter
        │    affine: rot ±12°, scale [0.9,1.1], shift ±2, shear_x ±0.15
        │    elastic: off by default (try α=1, σ=5 after shear A/B)
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
| Vertices 0–1023 | 32×32 bilinear upsample of the (possibly augmented) digit |
| Vertices 1024–2047 | 32×32 gradient magnitude of that plane, scaled to [-1, 1] |

Layout is **row-major blocks**, not a locality-preserving Hamming map. The goal is full occupancy plus a simple multi-view (ink ‖ edges), not spatial↔hypercube alignment. See [`docs/spatial_preprocess.md`](../docs/spatial_preprocess.md).

Test-set packing uses the **same** embed path with **no** augmentation (`HCNNSpatialAugConfig::None()`).

## Architecture

Topology and all other knobs live in **`DemoConfig`** near the top of `mnist_train.cpp` (layer list of `ArchLayer::Conv` / `Pool`, plus seed, LR, epochs, aug, …). Architecture scaffolding is shared with the regression demo in **`examples/demo_arch.h`** (`hcnn_demo::apply_arch` / `print_arch` / `summarize_arch`); param counts are checked against `HCNN::GetWeightCount`. `dim` also drives SpatialEmbed.

**Default** (`DemoConfig{}` field defaults):

```
Input: SpatialEmbed DualPlane 2048 floats (DIM=11, 1 channel)
  |
Conv1: 1  -> 16 channels, K=12 (DIM+1 self+neighbors), RELU, bias
Conv2: 16 -> 16 channels, K=12, RELU, bias
Conv3: 16 -> 16 channels, K=12, RELU, bias
  |
Readout: FLATTEN -> linear 32768->10 -> logits
```

Total parameters: **334,074** (208 conv1 + 3,088 conv2 + 3,088 conv3 + 327,690 readout).

To try another stack or schedule, edit `DemoConfig` fields at the top of the `.cpp` (layers, `dim`, `weight_seed`, `epochs`, `lr_max`, aug, …). Pools reduce DIM by 1 and shrink the FLATTEN head; BN is available per conv but is not the documented MNIST recipe.

**No antipodal pool (default)** — DIM stays 11 and N stays 2048 for all three convs, so the FLATTEN head sees every packed vertex (`32768→10`). Antipodal MAX pairs ink-half with grad-half indices on this pack and halves addressable positions; skipping pool is the default for this MNIST recipe. FLATTEN treats every (channel, vertex) activation as an independent feature.

**Headline result (seed `398479293`):** best-acc **99.46%** / best-loss CE **0.0155** (acc **99.45%**) under the default shear recipe — see Results.

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | Adam | Decoupled weight decay (AdamW), default betas (0.9, 0.999) |
| Learning rate | `lr_max = 0.001` | Peak on first epoch |
| LR schedule | Cosine `lr_max → 0.1·lr_max` | Floor **1e-4**; progress `epoch/(epochs-1)` hits `lr_min` on the last epoch |
| Batch size | 256 | Via `TrainEpoch` → `TrainBatch` |
| Weight decay | 1e-3 | Kernels + readout weights (not biases) |
| Epochs | **100** | |
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` (fixed stream; not varied with weight seed) |
| Weight init seed | **398479293** (default) | Printed as `Weight init seed:`; change `weight_seed` in `mnist_train.cpp` to probe init variance. Aug/shuffle seeds stay fixed. |
| **Augmentation** | train only | `HCNNSpatialAugmenter`: rot U[−12°, +12°]; scale U[0.9, 1.1]; shift ±2; **shear_x** U[−0.15, 0.15]; **elastic off** by default (`aug_elastic_alpha=0`; try α=1, σ=5 after shear A/B); noise σ=0.03; OOB = −1; **rebuilt every epoch**. Elastic cost O(H·W·⌈3σ⌉) when on. |
| Checkpoints | dual | Best test **loss** and best test **acc** via `HCNNDualCheckpoint`; net left on best-acc weights |

## Data loading

Raw IDX load via `load_mnist()` (`dataloader/HCNNDataset.h`) still returns 784-vectors. Augment + embed + flat buffers live in `examples/mnist_train.cpp` so the core loader stays format-only.

```cpp
auto train_raw = load_mnist("data/train-images-idx3-ubyte",
                            "data/train-labels-idx1-ubyte", 60000);
auto test_raw  = load_mnist("data/t10k-images-idx3-ubyte",
                            "data/t10k-labels-idx1-ubyte",  10000);
// fill_spatial_dataset: optional SpatialAug -> SpatialEmbed DualPlane -> FlatDataset
```

## Downloading the MNIST IDX files

The IDX files are not checked into the repository. Download them once and place them in `data/` at the project root:

```bash
mkdir -p data && cd data
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

After extraction:

```
data/train-images-idx3-ubyte    (45 MB)
data/train-labels-idx1-ubyte    (60 KB)
data/t10k-images-idx3-ubyte     (7.5 MB)
data/t10k-labels-idx1-ubyte     (10 KB)
```

The MNIST dataset is the work of Yann LeCun, Corinna Cortes, and Christopher J.C. Burges; this repository ships only the loader code.

## How to run

```bash
cmake --build cmake-build-release --target MNISTTrain
# MinGW runtime DLLs must be on PATH when running the exe
./cmake-build-release/MNISTTrain
```

## Results

60K train / 10K test, **DIM=11**, **3× Conv 16 RELU** (no antipodal pool), dense pack (32×32 ink ‖ 32×32 \|∇\|), train aug (**rot ±12°**, **scale [0.9, 1.1]**, shift ±2, **shear_x ±0.15**, elastic **off**, noise σ=0.03), Adam, batch=256, wd=1e-3, cosine LR **0.001 → 1e-4**, **100 epochs**. Kernel width **K = DIM + 1** (self + neighbors). Only the **weight init seed** varies; aug and shuffle streams are fixed. Epoch timing includes pack+aug rebuild + `TrainEpoch`. Throughput ~1.0–1.1k samples/s on 32 threads (~56–59 s/epoch). Parameters: **334,074**.

Documented headline numbers are the **default recipe (shear on)** under the current stack (self-tap conv + FLATTEN-only readout). Prefer quoting the **3-seed mean** for claims; quote a single seed only with the printed `Weight init seed`.

### Multi-seed (weight init only) — shear_x ±0.15

| Weight seed | Best acc | Loss @ best-acc | Best loss | Acc @ best-loss |
|-------------|----------|-----------------|-----------|-----------------|
| **398479293** (default) | **99.46%** | 0.01646 @ ep **60** | **0.01554** @ ep **81** | **99.45%** |
| **287821292** | **99.43%** | 0.01937 @ ep **98** | **0.01746** @ ep **71** | **99.42%** |
| **498279213** | **99.42%** | 0.01818 @ ep **81** | **0.01814** @ ep **97** | **99.39%** |

| Statistic | Value (3 seeds) |
|-----------|-----------------|
| **Mean best-acc** | **99.437%** |
| Range (best-acc) | **0.04 pp** (99.42–99.46) |
| Mean best-loss CE | **0.01705** |
| Mean acc @ best-loss | **99.420%** |

**Read:** init variance on peak accuracy is **tiny** (four hundredths of a point). All three seeds clear **99.4%** best-acc. Best-loss CE spans 0.0155–0.0181 (still ~99.4% acc@best-loss). Safe headline: **~99.44% mean best-acc** (3 weight seeds, same aug/shuffle streams).

### Checkpoints — seed 398479293 (default)

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | **81** | **0.01554** | **99.45%** (9945/10000) |
| **Best acc** | **60** | 0.01646 | **99.46%** (9946/10000) |

First ≥99% at epoch **16** (99.14%). Dual restore confirmed both snapshots (net left on best-acc).

### Checkpoints — seed 287821292

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | **71** | **0.01746** | **99.42%** (9942/10000) |
| **Best acc** | **98** | 0.01937 | **99.43%** (9943/10000) |

First ≥99% at epoch **19** (99.00%). Dual restore confirmed both snapshots.

### Checkpoints — seed 498279213

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | **97** | **0.01814** | **99.39%** (9939/10000) |
| **Best acc** | **81** | 0.01818 | **99.42%** (9942/10000) |

First ≥99% at epoch **20** (99.04%). Dual restore confirmed both snapshots. Dual checkpoints nearly coincide on CE (~0.0182).

### Curve (seed 398479293)

```
Epoch  Test Acc    Test Loss   LR
  1    96.92%      0.1155      0.00100
  8    98.80%      0.0373      0.00099
 16    99.14%      0.0270      0.00095   ← first ≥99%
 22    99.16%      0.0242      0.00090
 29    99.22%      0.0222      0.00083
 35    99.32%      0.0197      0.00076
 43    99.35%      0.0187      0.00066
 56    99.41%      0.0181      0.00047
 60    99.46%      0.0165      0.00042   ← best acc
 71    99.42%      0.0157      0.00028
 81    99.45%      0.0155      0.00018   ← best loss
100    99.31%      0.0177      0.00010
```

### Curve (seed 287821292)

```
Epoch  Test Acc    Test Loss   LR
  1    97.49%      0.0941      0.00100
 12    98.93%      0.0304      0.00097
 19    99.00%      0.0301      0.00093   ← first ≥99%
 26    99.27%      0.0243      0.00087
 55    99.28%      0.0208      0.00049
 66    99.37%      0.0192      0.00034
 71    99.42%      0.0175      0.00028   ← best loss
 98    99.43%      0.0194      0.00010   ← best acc
100    99.42%      0.0191      0.00010
```

### Curve (seed 498279213)

```
Epoch  Test Acc    Test Loss   LR
  1    96.51%      0.1208      0.00100
 12    98.94%      0.0333      0.00097
 20    99.04%      0.0292      0.00092   ← first ≥99%
 27    99.22%      0.0246      0.00086
 52    99.32%      0.0204      0.00053
 70    99.41%      0.0188      0.00029
 81    99.42%      0.0182      0.00018   ← best acc
 97    99.39%      0.0181      0.00010   ← best loss
100    99.35%      0.0197      0.00010
```

### Historical notes (do not mix with headline table)

Older tables in this file (shear A/B, channel-width ladder, pre-self ~99.28%) used earlier recipes (e.g. 60 epochs, pre-self kernels, or different activations). They remain useful as qualitative guidance only; re-run under the current `DemoConfig` before quoting.

## Analysis

### Why dense pack

Leaving vertices 784–2047 at zero wasted most of N at DIM=11. The 32×32 ‖ |∇| pack uses every input slot with ink + edge structure, without claiming that hypercube bit-flips equal 2D adjacency.

### Why no pool

Antipodal MAX is correct mathematically (winner-take-all backprop checks out) but a poor fit for this layout: every pair at DIM=11 straddles the ink half and the grad half, and pooling halves the FLATTEN head. Keeping full N=2048 lets both views stay addressable through all three convs into the linear readout.

### Why FLATTEN only (no GAP)

The SDK head is **FLATTEN linear only** (`num_features = c × N`). Global average
pool over vertices performed poorly in early MNIST experiments and is not part
of `HCNNReadout`. DualPlane is a row-major multi-view pack, not a locality-
preserving map, so the head is intentionally **position-addressable**.

### Why self-tap kernels (K = DIM + 1)

Neighbor-only conv lacked a center weight. With self + bit axes, the pack
climbs into the mid–high 99s (3-seed best-acc **99.42–99.46%**, mean
**~99.44%**) where earlier documented shear runs sat near **~99.3%**. Extra
cost is small (~1/DIM params per conv).

### Why three convs + full-N FLATTEN

Depth is cheap in parameters (~3k per mid 16-wide conv vs ~328k readout) but
costs wall time. Quality on this demo lives in **pack + last map + fat FLATTEN
+ aug**, not in antipodal downsampling or a giant last-channel map.

### Why augmentation

The FLATTEN readout is strongly position-addressable. Rotation, scale, shift,
and **shear** force the model off absolute vertex memorization for a given
stroke or slant. Affine is one inverse bilinear warp on the 28×28 plane before
packing. **Mild elastic** remains optional (off by default). Aug is train-only
so reported test numbers stay on clean IDX images. The **100-epoch** cosine
gives room to finish climbing after the first ≥99%.

### Curve shape

Fast start (~97% after epoch 1), first ≥99% by ep **16–20**, then a long
mid-run climb into the high 99s. Dual-checkpoint timing **varies by seed**
(best-acc can land mid-run or late; best-loss often near best-acc on CE).
Across three seeds, best-acc range is only **0.04 pp** — init noise is small
relative to the jump from pre-self ~99.3%.

### What ~99.4% means

- Linear classifier on raw pixels ~92%
- 2-layer MLP ~98%
- Spatial 2D CNNs typically 99.0–99.5%+

HypercubeCNN at **~99.44% mean best-acc** (3 seeds; range **99.42–99.46%**)
sits in **solid spatial-CNN territory** without a grid prior (row-major pack +
Hamming self/neighbor kernels + FLATTEN). Remaining errors ~54–58/10k at
best-acc. Optional elastic is a secondary lever — not antipodal pooling on
this demo pack.

## Significance

**~99.44% mean best-acc** (3 weight seeds: **99.46% / 99.43% / 99.42%**) for a
**3-conv** no-pool HypercubeCNN (DIM=11, DualPlane pack, shear_x train aug,
K=DIM+1, 100 epochs) shows the stack is demo-solid and that **self-tap
kernels + depth + full-N FLATTEN + invariance-inducing aug** matter when the
input is an engineered image embedding rather than native hypercube data.
Pack, aug, schedule, depth, and reported weight seeds are **image-demo
engineering**, documented here separately from the core SDK (fingerprints,
Boolean functions, reservoir / ESN state).
