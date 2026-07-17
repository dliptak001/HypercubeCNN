# MNIST Training -- Handwritten Digit Classification

Demonstrates end-to-end training and evaluation of a HypercubeCNN on the MNIST handwritten digit dataset (60K train, 10K test, 10 classes).

## What this example shows

- Loading real MNIST data from IDX binary files
- **Dense input pack** for DIM=11: full N=2048 occupancy (no zero pad)
- **Train-time augmentation**: shift ±2 px + light Gaussian noise
- 2-stage conv+pool architecture with FLATTEN readout
- Mini-batch Adam, cosine LR annealing, weight decay
- Dual checkpoints: best test loss and best test accuracy (`GetWeights` / `SetWeights`)
- Parallel batch inference for evaluation

## How MNIST maps onto the hypercube

### Loader

MNIST images are 28×28 = 784 grayscale pixels, normalized to **[-1.0, 1.0]** (background ≈ −1).

### Dense pack (always length 2048)

Before `TrainEpoch` / `ForwardBatch`, each image is packed into a **full** hypercube input (N = 2^11 = 2048). There is **no** zero-padding of unused vertices.

```
28×28 digit in [-1, 1]
        │
        │  (train only) shift ±2 px, N(0, 0.03²) noise, clip to [-1, 1]
        ▼
 bilinear resize → 32×32 image
        │
        ├──► out[0 .. 1023]      = 32×32 image (row-major)
        │
        └──► finite-diff |∇| → per-image max-norm → [-1, 1]
                    │
                    └──► out[1024 .. 2047] = 32×32 |∇| (row-major)
```

| Region | Content |
|--------|---------|
| Vertices 0–1023 | 32×32 bilinear upsample of the (possibly augmented) digit |
| Vertices 1024–2047 | 32×32 gradient magnitude of that plane, scaled to [-1, 1] |

Layout is **row-major blocks**, not a locality-preserving Hamming map. The goal is full occupancy plus a simple multi-view (ink ‖ edges), not spatial↔hypercube alignment.

Test-set packing uses the **same** transform with **no** augmentation.

## Architecture

```
Input: dense pack 2048 floats (DIM=11, 1 channel)
  |
Conv1: 1 -> 16 channels, K=11, ReLU, bias
Pool1: MAX (antipodal), DIM 11->10, N 2048->1024
  |
Conv2: 16 -> 16 channels, K=10, ReLU, bias
  |
Readout: FLATTEN -> linear 16384->10 -> logits
```

Total parameters: **166,618** (192 conv1 + 2,576 conv2 + 163,850 readout).

The readout dominates (~98% of params). FLATTEN treats every (channel, vertex) activation as an independent feature.

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | Adam | Decoupled weight decay (AdamW), default betas (0.9, 0.999) |
| Learning rate | `lr_max = 0.001` | Peak on first epoch |
| LR schedule | Cosine `lr_max → 0.1·lr_max` | Floor **1e-4**; progress `epoch/(epochs-1)` hits `lr_min` on the last epoch |
| Batch size | 256 | Via `TrainEpoch` → `TrainBatch` |
| Weight decay | 1e-3 | Kernels + readout weights (not biases) |
| Epochs | 40 | |
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` (fixed stream; not varied with weight seed) |
| Weight init seed | **398479293** (default) | Printed as `Weight init seed:`; change `weight_seed` in `mnist_train.cpp` for multi-seed runs. Aug/shuffle seeds stay fixed. See multi-seed table below. |
| **Augmentation** | train only | Independent shift \(dx,dy \in \{-2,\ldots,2\}\); Gaussian noise σ=0.03; border fill = −1; **rebuilt every epoch** |
| Checkpoints | dual | Best test **loss** and best test **acc**; net left on best-acc weights |

## Data loading

Raw IDX load via `load_mnist()` (`dataloader/HCNNDataset.h`) still returns 784-vectors. Packing and aug live in `examples/mnist_train.cpp` so the core loader stays format-only.

```cpp
auto train_raw = load_mnist("data/train-images-idx3-ubyte",
                            "data/train-labels-idx1-ubyte", 60000);
auto test_raw  = load_mnist("data/t10k-images-idx3-ubyte",
                            "data/t10k-labels-idx1-ubyte",  10000);
// fill_packed_dataset(...): 28×28 → 2048; aug optional
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

60K train / 10K test, **DIM=11**, dense pack (32×32 image ‖ 32×32 \|∇\|), train aug (shift ±2, σ=0.03), Adam, batch=256, wd=1e-3, cosine LR **0.001 → 1e-4**, 40 epochs. Only the **weight init seed** varies across runs; aug and shuffle streams are fixed. Epoch timing includes pack+aug rebuild + `TrainEpoch`.

### Multi-seed (weight init only)

| Weight seed | Best acc | Loss @ best-acc | Best loss | Acc @ best-loss |
|-------------|----------|-----------------|-----------|-----------------|
| 42 | 98.56% | 0.0468 | 0.0450 | 98.55% |
| 983247375 | 98.68% | 0.0419 | 0.0415 | 98.65% |
| **398479293** | **98.71%** | **0.0410** | **0.0401** | **98.68%** |

| Statistic (best-acc) | Value |
|----------------------|--------|
| Mean | **98.65%** |
| Range | 98.56% – 98.71% (0.15 pp) |
| Mean best-loss CE | **~0.042** |

Init seed moves accuracy by about **±0.1 pp** on this recipe — real but small. Prefer quoting **mean ~98.65% over seeds** for claims; quote a single seed only with the printed `Weight init seed`.

**Documented default seed:** `398479293` (best of the three). Throughput ~3.3–3.6k samples/s on 32 threads (~17 s/epoch).

### Representative curve (seed 398479293)

```
Epoch  Test Acc    Test Loss   LR
  1    95.03%      0.1981      0.00100
 12    98.12%      0.0609      0.00083
 22    98.49%      0.0468      0.00050
 30    98.68%      0.0424      0.00024
 37    98.71%      0.0410      0.00011   ← best acc
 40    98.68%      0.0401      0.00010   ← best loss
```

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | 40 | **0.0401** | 98.68% |
| **Best acc** | 37 | 0.0410 | **98.71%** |

### vs earlier DIM=11 baselines

| Recipe | Best loss | Best acc |
|--------|-----------|----------|
| Zero-pad 784→2048, no aug (seed 42) | 0.0676 @ 98.00% | 98.27% (loss 0.094) |
| Dense pack + aug (3 seeds, mean) | ~0.042 | **~98.65%** |
| Dense pack + aug, best seed (398479293) | **0.0401 @ 98.68%** | **98.71%** |

Pack + aug removed the early loss minimum / late CE rise pattern. Best-loss and best-acc stay close across seeds.

## Analysis

### Why dense pack

Leaving vertices 784–2047 at zero wasted most of N at DIM=11. The 32×32 ‖ |∇| pack uses every input slot with ink + edge structure, without claiming that hypercube bit-flips equal 2D adjacency.

### Why augmentation

The FLATTEN readout is strongly position-addressable. Small shifts force the model to rely less on absolute vertex indices for a given stroke. Light noise softens intensity memorization. Aug is train-only so reported test numbers stay on clean IDX images.

### Curve shape

Test **loss trends down through the run** (still improving near epoch 40) rather than bottoming early and climbing. Dual checkpoints still run; with this recipe they almost pick the same model.

### What ~98.7% means

- Linear classifier on raw pixels ~92%
- 2-layer MLP ~98%
- Spatial 2D CNNs 99.0–99.5%

HypercubeCNN at **~98.65% mean / 98.71% best seed** is still in **MLP / weak-CNN territory without a spatial grid prior** (row-major blocks + antipodal pool). Remaining errors ~129–144/10k; 99%+ usually needs stronger spatial structure or heavier aug. Pack + aug closed most of the avoidable gap from empty pad and absolute-position overfitting.

## Significance

**~98.65% mean test accuracy** (three weight-init seeds; peak **98.71%**) with best-loss CE around **0.04** on MNIST, without spatial grid convolutions, shows the training stack is solid for demos and for **native hypercube data** (fingerprints, Boolean functions, reservoir state). Dense pack, aug, and reported weight seeds are **image-demo engineering**, documented here separately from the core SDK.
