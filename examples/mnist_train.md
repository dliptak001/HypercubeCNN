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
Conv1: 1  -> 16 channels, K=12 (DIM+1 self+neighbors), NONE, bias
Conv2: 16 -> 16 channels, K=12, TANH, bias
Conv3: 16 -> 16 channels, K=12, RELU, bias
  |
Readout: FLATTEN -> linear 32768->10 -> logits
```

Total parameters: **334,082** (208 conv1 + 3,092 conv2 + 3,092 conv3 + 327,690 readout).

To try another stack or schedule, edit `DemoConfig` fields at the top of the `.cpp` (layers, `dim`, `weight_seed`, `epochs`, `lr_max`, aug, …). Pools reduce DIM by 1 and shrink the FLATTEN head; BN is available per conv but is not the documented MNIST recipe.

**No antipodal pool (default)** — DIM stays 11 and N stays 2048 for all three convs, so the FLATTEN head sees every packed vertex (`32768→10`). Antipodal MAX pairs ink-half with grad-half indices on this pack and halves addressable positions; skipping pool is the default for this MNIST recipe. FLATTEN treats every (channel, vertex) activation as an independent feature.

**Depth:** On seed `398479293` (pre-shear), the **3-conv** stack beats the prior **2-conv** recipe (best-acc **99.31%** vs **99.27%**). With default shear, same seed peaks at **99.28%** best-acc (see Results).

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

60K train / 10K test, **DIM=11**, **3× Conv 16** NONE→TANH→RELU (no antipodal pool), dense pack (32×32 image ‖ 32×32 \|∇\|), train aug (**rot ±12°**, **scale [0.9, 1.1]**, shift ±2, **shear_x ±0.15**, elastic **off**, noise σ=0.03), Adam, batch=256, wd=1e-3, cosine LR **0.001 → 1e-4**, **60 epochs**. Only the **weight init seed** varies; aug and shuffle streams are fixed. Epoch timing includes pack+aug rebuild + `TrainEpoch`. Throughput ~1.09–1.11k samples/s on 32 threads (~54–56 s/epoch). Parameters: **334,082** (K=DIM+1 self+neighbors). Accuracy figures below were measured **before** the self tap (`pre_self_contribution` tag); re-run before quoting post-self numbers.

Documented numbers below are the **default recipe including shear_x** (elastic off). Same weight seeds as the pre-shear table for fair multi-seed fill-in.

### Multi-seed (weight init only) — with shear_x ±0.15

| Weight seed | Best acc | Loss @ best-acc | Best loss | Acc @ best-loss |
|-------------|----------|-----------------|-----------|-----------------|
| **398479293** | **99.28%** | 0.0216 @ ep 56 | **0.0210** @ ep 60 | 99.25% |
| 287821292 | *TBD* | *TBD* | *TBD* | *TBD* |
| 498279213 | *TBD* | *TBD* | *TBD* | *TBD* |

| Statistic (best-acc) | Value |
|----------------------|--------|
| Mean (3 seeds) | *TBD* (1/3 filled) |
| Range | *TBD* |
| Mean best-loss CE | *TBD* |

Set `weight_seed` in `DemoConfig` to each of the two open seeds and fill the row + checkpoint block after the run. Prefer quoting a multi-seed **mean** once all three are filled; quote a single seed only with the printed `Weight init seed`.

### A/B — seed 398479293: shear off vs on

Same arch/schedule/aug except `aug_shear_x_max` (0 vs 0.15). Elastic off both runs.

| Recipe | Best acc | Best loss | Acc @ best-loss | First ≥99% |
|--------|----------|-----------|-----------------|------------|
| No shear (prior) | **99.31%** @ ep 53 | 0.02135 @ ep 60 | 99.23% | ep 27 |
| **Shear_x ±0.15 (default)** | 99.28% @ ep 56 | **0.02099** @ ep 60 | **99.25%** | ep **22** |

**Read:** single-seed **wash on peak acc** (−0.03 pp, noise-scale); **slightly better best-loss CE**; earlier climb into the 99s. Keep shear as default pending more seeds; do not claim a mean win yet.

### Ablation — channel width (speed/quality ladder)

Same recipe as default shear run (seed `398479293`, RELU, no pool, elastic off);
only the conv channel list changes. When **last map = 16**, FLATTEN is
**32768→10** (~328k head). **8→16→32** doubles the head (**65536→10**).

| Stack | Params | Best acc | Best loss | Acc @ best-loss | Throughput |
|-------|--------|----------|-----------|-----------------|------------|
| **16→16→16** (default) | 334,082 | **99.28%** @ ep 56 | **0.02099** @ ep 60 | 99.25% | ~1.1k samples/s (~55 s/ep) |
| **16→8→16** | 330,722 | 99.22% @ ep 35 | 0.02170 @ ep 60 | 99.19% | ~1.9–2.0k samples/s (~30 s/ep) |
| **4→8→16** | 329,522 | 98.96% @ ep 58 | 0.03160 @ ep 58 | 98.96% | ~2.5–2.7k samples/s (~23 s/ep) |
| **8→16→32** | 662,554 | 99.25% @ ep 60 | 0.02221 @ ep 60 | 99.25% | ~0.82k samples/s (~73 s/ep) |

**Read:**

- **16→8→16** is an **admirable speed/quality trade**: ~**2×** wall throughput
  for about **−0.06 pp** best-acc and only a slight CE hit. Best mid-width
  default for sweeps / interactive demos.
- **4→8→16** is **not horrible**: still **~98.96%** best-acc at ~**2.4–2.5×**
  default speed. Cost is mostly **CE** (best-loss 0.032 vs 0.021) and never
  quite clearing a clean 99% on this seed. Dual checkpoints coincide (ep 58).
  Fine for smoke / “does train work?”; not the accuracy recipe.
- **8→16→32** (CNN-style widen + **2× head**) is a **bad deal** here: **−0.03 pp**
  best-acc vs default, worse CE, **~2× params**, **~1.3× slower**. Without
  spatial downsampling, last=32 mostly fattens FLATTEN memorization capacity;
  it does not buy accuracy on this seed. Skip for the demo recipe.
- Story: quality lives in **pack + last map + fat FLATTEN**; thinning early/mid
  width is mostly free speed; **widening the last map** is costly and not
  helpful. Keep **16→16→16** as the documented accuracy default; use
  **16→8→16** when iterating.

### Checkpoints — seed 398479293 (default, shear on)

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | 60 | **0.02099** | 99.25% (9925/10000) |
| **Best acc** | 56 | 0.02157 | **99.28%** (9928/10000) |

First ≥99% at epoch **22** (99.06%). Dual restore confirmed both snapshots.

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

### Curve (seed 398479293, shear on)

```
Epoch  Test Acc    Test Loss   LR
  1    96.76%      0.1166      0.00100
 12    98.82%      0.0354      0.00092
 22    99.06%      0.0294      0.00075   ← first ≥99%
 26    99.14%      0.0267      0.00066
 35    99.16%      0.0248      0.00044
 47    99.23%      0.0227      0.00020
 51    99.23%      0.0219      0.00015
 56    99.28%      0.0216      0.00011   ← best acc
 60    99.25%      0.0210      0.00010   ← best loss
```

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

### Why three convs

On seed `398479293` (pre-shear recipe), adding a third 16-wide RELU conv (same
DIM, same FLATTEN head size) lifts best-acc from **99.27%** (2-conv) to
**99.31%** and best-loss CE from ~0.024 to **~0.021**. Extra depth is cheap in
parameters (~2.8k for the third conv vs 328k readout) but costs wall time
(~1.1k vs ~1.9k samples/s).

Channel width ablations on this seed (see Results): mid **16→8→16** still
clears **99.2%** at ~**2×** samples/s; **4→8→16** hits **~99.0%** at ~**2.5×**;
CNN-style **8→16→32** (2× head) does **not** beat 16-wide and is slower —
skip. Prefer thinning early/mid width over fattening the last map when wall
time matters.

### Why augmentation

The FLATTEN readout is strongly position-addressable. Rotation, scale, shift,
and **shear** force the model off absolute vertex memorization for a given
stroke or slant. Affine is one inverse bilinear warp on the 28×28 plane before
packing. On seed `398479293`, **shear_x ±0.15** vs no shear is a **wash on
peak acc** (−0.03 pp) with a **slightly better best-loss CE** and earlier
first ≥99% (ep 22 vs 27). **Mild elastic** remains optional (off by default);
it is a second smooth displacement pass and usually dominates pack+aug wall
time. Aug is train-only so reported test numbers stay on clean IDX images.
Stronger geometry needs the full **60-epoch** cosine to finish climbing into
the high 99s.

### Curve shape

Fast start (~97% after epoch 1), then a long climb under cosine decay. With
shear, first ≥99% lands earlier (ep **22**); best acc mid–late (ep **56**);
best loss at the final epoch (**60**). Dual checkpoints stay within ~0.0006 CE
on this seed. Multi-seed variance for the shear recipe is not yet measured.

### What ~99.3% means

- Linear classifier on raw pixels ~92%
- 2-layer MLP ~98%
- Spatial 2D CNNs typically 99.0–99.5%+

HypercubeCNN at **99.28% best-acc** (seed 398479293; best-loss CE **0.021**,
default **shear_x** recipe) sits in **light spatial-CNN territory** without a
grid prior (row-major pack + Hamming kernels + FLATTEN). Remaining errors
~72/10k at best-acc. Other levers: more weight seeds for a mean claim,
elastic, TANH–TANH–RELU activation A/B, locality-aware packing — not antipodal
pooling on this demo pack.

## Significance

**99.28% best-acc / 99.25% at best-loss** on the documented default weight seed
for a **3-conv** no-pool HypercubeCNN (DIM=11, dense pack, shear_x train aug)
shows the training stack is solid for demos and that **depth + full-N FLATTEN
+ invariance-inducing aug** matter when the input is an engineered image
embedding rather than native hypercube data. Pack, aug, schedule, depth, and
reported weight seed are **image-demo engineering**, documented here
separately from the core SDK (fingerprints, Boolean functions, reservoir
state).
