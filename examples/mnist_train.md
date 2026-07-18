# MNIST Training -- Handwritten Digit Classification

Demonstrates end-to-end training and evaluation of a HypercubeCNN on the MNIST handwritten digit dataset (60K train, 10K test, 10 classes).

## What this example shows

- Loading real MNIST data from IDX binary files
- **Core spatial preprocess**: `HCNNSpatialAugmenter` (train) → `HCNNSpatialEmbedder` DualPlaneResize (DIM=11, full N=2048)
- **Train-time augmentation**: rotate ±12°, scale [0.9, 1.1], shift ±2 px, light Gaussian noise
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
        │    rot ±12°, scale [0.9,1.1], shift ±2, N(0, 0.03²), clip after noise
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
Conv1: 1  -> 16 channels, K=11, ReLU, bias
Conv2: 16 -> 16 channels, K=11, ReLU, bias
Conv3: 16 -> 16 channels, K=11, ReLU, bias
  |
Readout: FLATTEN -> linear 32768->10 -> logits
```

Total parameters: **333,546** (192 conv1 + 2,832 conv2 + 2,832 conv3 + 327,690 readout).

To try another stack or schedule, edit `DemoConfig` fields at the top of the `.cpp` (layers, `dim`, `weight_seed`, `epochs`, `lr_max`, aug, …). Pools reduce DIM by 1 and shrink the FLATTEN head; BN is available per conv but is not the documented MNIST recipe.

**No antipodal pool (default)** — DIM stays 11 and N stays 2048 for all three convs, so the FLATTEN head sees every packed vertex (`32768→10`). Antipodal MAX pairs ink-half with grad-half indices on this pack and halves addressable positions; skipping pool is the default for this MNIST recipe. FLATTEN treats every (channel, vertex) activation as an independent feature.

**Depth:** On the documented default weight seed, the **3-conv** stack beats the prior **2-conv** recipe (best-acc **99.31%** vs **99.27%** on seed `398479293`).

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | Adam | Decoupled weight decay (AdamW), default betas (0.9, 0.999) |
| Learning rate | `lr_max = 0.001` | Peak on first epoch |
| LR schedule | Cosine `lr_max → 0.1·lr_max` | Floor **1e-4**; progress `epoch/(epochs-1)` hits `lr_min` on the last epoch |
| Batch size | 256 | Via `TrainEpoch` → `TrainBatch` |
| Weight decay | 1e-3 | Kernels + readout weights (not biases) |
| Epochs | 60 | |
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` (fixed stream; not varied with weight seed) |
| Weight init seed | **398479293** (default) | Printed as `Weight init seed:`; change `weight_seed` in `mnist_train.cpp` to probe init variance. Aug/shuffle seeds stay fixed. |
| **Augmentation** | train only | `HCNNSpatialAugmenter`: rot U[−12°, +12°]; scale U[0.9, 1.1] about center; shift dx,dy in {−2,…,2}; Gaussian noise σ=0.03; OOB = −1; single bilinear warp; **rebuilt every epoch** |
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

60K train / 10K test, **DIM=11**, **3× Conv 16** (no antipodal pool), dense pack (32×32 image ‖ 32×32 \|∇\|), train aug (**rot ±12°**, **scale [0.9, 1.1]**, shift ±2, σ=0.03), Adam, batch=256, wd=1e-3, cosine LR **0.001 → 1e-4**, **60 epochs**. Only the **weight init seed** varies; aug and shuffle streams are fixed. Epoch timing includes pack+aug rebuild + `TrainEpoch`. Throughput ~1.1k samples/s on 32 threads (~54–56 s/epoch). Parameters: **333,546**.

Prior 2-conv multi-seed tables are **retired**. Same weight seeds as before for a fair re-test of the 3-conv stack.

### Multi-seed (weight init only)

| Weight seed | Best acc | Loss @ best-acc | Best loss | Acc @ best-loss |
|-------------|----------|-----------------|-----------|-----------------|
| **398479293** | **99.31%** | 0.0218 @ ep 53 | **0.0213** @ ep 60 | 99.23% |
| 287821292 | *TBD* | *TBD* | *TBD* | *TBD* |
| 498279213 | *TBD* | *TBD* | *TBD* | *TBD* |

| Statistic (best-acc) | Value |
|----------------------|--------|
| Mean (3 seeds) | *TBD* (1/3 filled) |
| Range | *TBD* |
| Mean best-loss CE | *TBD* |

Set `weight_seed` in `DemoConfig` to each of the two open seeds and fill the row + checkpoint block after the run. Prefer quoting a multi-seed **mean** once all three are filled; quote a single seed only with the printed `Weight init seed`.

### Checkpoints — seed 398479293 (default)

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | 60 | **0.02135** | 99.23% (9923/10000) |
| **Best acc** | 53 | 0.02177 | **99.31%** (9931/10000) |

First ≥99% at epoch **27** (99.09%). Dual restore confirmed both snapshots.

### Checkpoints — seed 287821292

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | *TBD* | *TBD* | *TBD* |
| **Best acc** | *TBD* | *TBD* | *TBD* |

First ≥99%: *TBD*.

### Checkpoints — seed 498279213

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | *TBD* | *TBD* | *TBD* |
| **Best acc** | *TBD* | *TBD* | *TBD* |

First ≥99%: *TBD*.

### Curve (seed 398479293)

```
Epoch  Test Acc    Test Loss   LR
  1    96.99%      0.1056      0.00100
 12    98.73%      0.0388      0.00092
 22    98.95%      0.0331      0.00075
 27    99.09%      0.0270      0.00063   ← first ≥99%
 33    99.13%      0.0278      0.00049
 47    99.27%      0.0225      0.00020   ← joint best-loss + best-acc (then)
 50    99.21%      0.0217      0.00016
 53    99.31%      0.0218      0.00013   ← best acc
 60    99.23%      0.0213      0.00010   ← best loss
```

## Analysis

### Why dense pack

Leaving vertices 784–2047 at zero wasted most of N at DIM=11. The 32×32 ‖ |∇| pack uses every input slot with ink + edge structure, without claiming that hypercube bit-flips equal 2D adjacency.

### Why no pool

Antipodal MAX is correct mathematically (winner-take-all backprop checks out) but a poor fit for this layout: every pair at DIM=11 straddles the ink half and the grad half, and pooling halves the FLATTEN head. Keeping full N=2048 lets both views stay addressable through all three convs into the linear readout.

### Why three convs

On seed `398479293`, adding a third 16-wide RELU conv (same DIM, same FLATTEN head size) lifts best-acc from **99.27%** (2-conv) to **99.31%** and best-loss CE from ~0.024 to **~0.021**. Extra depth is cheap in parameters (~2.8k for the third conv vs 328k readout) but costs wall time (~1.1k vs ~1.9k samples/s).

### Why augmentation

The FLATTEN readout is strongly position-addressable. Rotation and mild scale (plus shift and noise) force the model off absolute vertex memorization for a given stroke. Geometry is one inverse bilinear warp on the 28×28 plane before packing. Aug is train-only so reported test numbers stay on clean IDX images. Stronger geometry needed a longer cosine (**60 epochs**) to finish climbing into the high 99s.

### Curve shape

Fast start (~97% after epoch 1), then a long climb under cosine decay. Best acc lands mid–late schedule (ep **53**); best loss is at the final epoch (**60**). Dual checkpoints stay within ~0.0004 CE on this seed. Multi-seed variance for 3-conv is not yet re-measured.

### What ~99.3% means

- Linear classifier on raw pixels ~92%
- 2-layer MLP ~98%
- Spatial 2D CNNs typically 99.0–99.5%+

HypercubeCNN at **99.31% best-acc** (seed 398479293; best-loss CE **0.021**) sits in **light spatial-CNN territory** without a grid prior (row-major pack + Hamming kernels + FLATTEN). Remaining errors ~69/10k at best-acc. Optional next levers: more weight seeds for a mean claim, elastic/shear aug, or locality-aware packing — not antipodal pooling on this demo pack.

## Significance

**99.31% best-acc / 99.23% at best-loss** on the documented default weight seed for a **3-conv** no-pool HypercubeCNN (DIM=11, dense pack, geometric train aug) shows the training stack is solid for demos and that **depth + full-N FLATTEN + invariance-inducing aug** matter when the input is an engineered image embedding rather than native hypercube data. Pack, aug, schedule, depth, and reported weight seed are **image-demo engineering**, documented here separately from the core SDK (fingerprints, Boolean functions, reservoir state).
