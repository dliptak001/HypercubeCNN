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
| Shuffle | per-epoch | `shuffle_seed = epoch + 1` |
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

60K train / 10K test, **DIM=11**, dense pack (32×32 image ‖ 32×32 \|∇\|), train aug (shift ±2, σ=0.03), Adam, batch=256, wd=1e-3, cosine LR **0.001 → 1e-4**, 40 epochs. Epoch timing includes pack+aug rebuild + `TrainEpoch`.

```
Epoch  Test Acc    Test Loss   LR        Time/epoch
  1    94.46%      0.2245      0.00100   ~17s
  6    97.49%      0.0849      0.00096   ~17s
 16    97.96%      0.0636      0.00071   ~17s
 20    98.13%      0.0584      0.00057   ~18s
 26    98.50%      0.0524      0.00036   ~17s
 30    98.50%      0.0495      0.00024   ~17s
 35    98.56%      0.0468      0.00014   ~17s   ← best acc
 40    98.55%      0.0450      0.00010   ~17s   ← best loss
```

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | 40 | **0.0450** | 98.55% |
| **Best acc** | 35 | 0.0468 | **98.56%** |

**Throughput: ~3.3–3.5k samples/s** on 32 threads (~17 s/epoch wall-clock including pack+aug).

### vs prior DIM=11 baseline (zero-pad, no aug)

Same net and optimizer knobs; only embed + train aug differ:

| | Zero-pad, no aug | Dense pack + aug | Δ |
|--|------------------|------------------|---|
| Best loss | 0.0676 @ 98.00% | **0.0450 @ 98.55%** | **−33% CE**, +0.55% acc |
| Best acc | 98.27% (loss 0.094) | **98.56%** (loss 0.047) | **+0.29%** (~29 samples) |
| Loss @ best-acc | 0.094 | **0.047** | ~**2× better calibrated** |
| Best-loss vs best-acc | Split (ep 9 vs 37) | **Almost the same model** (ep 40 vs 35) | Dual checkpoints still useful, barely diverge |

## Analysis

### Why dense pack

Leaving vertices 784–2047 at zero wasted most of N at DIM=11. The 32×32 ‖ |∇| pack uses every input slot with ink + edge structure, without claiming that hypercube bit-flips equal 2D adjacency.

### Why augmentation

The FLATTEN readout is strongly position-addressable. Small shifts force the model to rely less on absolute vertex indices for a given stroke. Light noise softens intensity memorization. Aug is train-only so reported test numbers stay on clean IDX images.

### Curve shape (what changed)

With zero-pad and no aug, test **loss bottomed early** (~ep 9) then **rose** while accuracy crept up — classic overconfidence. With dense pack + aug, **loss keeps falling through epoch 40** and best-loss / best-acc nearly coincide (~98.55–98.56%, CE ~0.045–0.047). That is a healthier training dynamic, not just a higher peak.

### What ~98.6% means

- Linear classifier on raw pixels ~92%
- 2-layer MLP ~98%
- Spatial 2D CNNs 99.0–99.5%

HypercubeCNN at **98.56%** is still in **MLP / weak-CNN territory without a spatial grid prior** (row-major blocks + antipodal pool). The remaining gap to 99%+ is the cost of that design. Pack + aug closed a large fraction of the avoidable gap from wasted input slots and absolute-position overfitting.

## Significance

**98.56%** test accuracy (best-acc) with best loss **0.045** on MNIST, without spatial grid convolutions, shows the training stack is solid for demos and for **native hypercube data** (fingerprints, Boolean functions, reservoir state). Dense pack and aug are **image-demo engineering**, documented here separately from the core SDK.
