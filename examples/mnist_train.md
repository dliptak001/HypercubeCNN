# MNIST Training -- Handwritten Digit Classification

Demonstrates end-to-end training and evaluation of a HypercubeCNN on the MNIST handwritten digit dataset (60K train, 10K test, 10 classes).

## What this example shows

- Loading real MNIST data from IDX binary files
- Constructing a 4-stage conv+pool architecture
- Training with mini-batch SGD, momentum, cosine LR annealing, and weight decay
- Parallel batch inference for evaluation

## How MNIST maps onto the hypercube

MNIST images are 28x28 = 784 grayscale pixels. The network uses DIM=10, giving N = 2^10 = 1024 vertices. Pixels are normalized to [-1.0, 1.0] and assigned to vertices 0-783 via Direct Linear Assignment. Vertices 784-1023 are zero-padded.

No spatial locality is encoded -- the 2D pixel grid is flattened and mapped onto hypercube vertices in index order. The network must learn all useful relationships from the hypercube topology alone.

## Architecture

```
Input: 784 pixels -> 1024 vertices (DIM=10)
  |
Conv1: 1 -> 32 channels, K=10, ReLU, bias
Pool1: MAX, DIM 10->9, N 1024->512
  |
Conv2: 32 -> 64 channels, K=9, ReLU, bias
Pool2: MAX, DIM 9->8, N 512->256
  |
Conv3: 64 -> 128 channels, K=8, ReLU, bias
Pool3: MAX, DIM 8->7, N 256->128
  |
Conv4: 128 -> 128 channels, K=7, ReLU, bias
Pool4: MAX, DIM 7->6, N 128->64
  |
Readout: GAP per channel -> linear 128->10 -> logits
```

Total parameters: ~200K.

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | SGD + momentum | Standard momentum SGD with L2 weight decay |
| Learning rate | 0.06 | Initial LR, decays via cosine annealing |
| LR schedule | Cosine annealing to 1e-5 | Smooth decay, no warmup or restarts |
| Momentum | 0.9 | |
| Batch size | 256 | Parallel across threads via `TrainBatch` (dispatched by `TrainEpoch`) |
| Weight decay | 1e-4 | Applied to kernels and readout weights (not biases) |
| Epochs | 40 | |
| Shuffle | per-epoch | `TrainEpoch(..., shuffle_seed = epoch + 1)` |

## Data loading

MNIST data is loaded from IDX binary files using `load_mnist()` from `dataloader/HCNNDataset.h`:

```cpp
auto train_data = load_mnist("data/train-images-idx3-ubyte",
                             "data/train-labels-idx1-ubyte", 20000);
auto test_data  = load_mnist("data/t10k-images-idx3-ubyte",
                             "data/t10k-labels-idx1-ubyte",  2000);
```

The `HCNNDataset` struct holds a vector of `Sample` (each with `std::vector<float> input` and `int target_class`). The example flattens these into a contiguous `FlatDataset` buffer (one `float[]` for all inputs, one `int[]` for all targets) and feeds it to `HCNN::TrainEpoch` (which handles shuffling and batching).

The shipped example loads a 20K / 2K subset for fast iteration. Bump both calls to `60000` / `10000` to train on the full dataset (see [Benchmark results](#benchmark-results) below).

## Downloading the MNIST IDX files

The IDX files are not checked into the repository. Download them once from any of the following mirrors and place them in `data/` at the project root:

```bash
mkdir -p data && cd data
# Mirror 1: Tensorflow
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

After extraction the directory should contain four files:

```
data/train-images-idx3-ubyte    (45 MB)
data/train-labels-idx1-ubyte    (60 KB)
data/t10k-images-idx3-ubyte     (7.5 MB)
data/t10k-labels-idx1-ubyte     (10 KB)
```

The MNIST dataset is the work of Yann LeCun, Corinna Cortes, and Christopher J.C. Burges and is distributed under their original terms; the HypercubeCNN repository ships only the loader code.

## How to run

Build and run (requires MinGW on PATH for runtime DLLs):

```bash
cmake --build cmake-build-release
./cmake-build-release/MNISTTrain
```

## Expected output

The shipped example loads a 20K / 2K subset for fast iteration:

```
Loading MNIST from .../data...
Train: 20000 samples, Test: 2000 samples
Threads: 32

=== HCNN (lr=0.06, batch=256, wd=0.0001) ===
Initial test: loss=2.302... acc=196/2000 (9.8%)
Epoch 1/40: loss=0.35... acc=1780/2000 (89.0%)
  (lr=0.059..., 1.1s)
...
Epoch 40/40: loss=0.06... acc=1920/2000 (96.0%)
  (lr=0.00001, 1.0s)
```

Exact numbers vary by platform and thread count. Bump `load_mnist` counts to `60000` / `10000` for the full dataset (~98% accuracy after 20-40 epochs -- see [Benchmark results](#benchmark-results) below).

## Benchmark results

Full-dataset runs (60K train / 10K test). The architecture and optimizer are as described above; only batch size and epoch count vary.

### Run 1 -- batch=32, 20 epochs

```
Epoch  Test Acc   Test Loss   LR        Time
  1    93.99%     0.1940      0.0600    270s
  2    94.60%     0.1698      0.0599    275s
  3    96.21%     0.1240      0.0596    266s
  6    97.31%     0.0828      0.0577    278s
 10    97.98%     0.0700      0.0528    270s
 15    97.90%     0.0691      0.0436    272s
 19    98.10%     0.0629      0.0347    272s
 20    98.08%     0.0646      0.0324    277s
```

**Peak accuracy: 98.10%** (epoch 19)

### Run 2 -- batch=256, 40 epochs

```
Epoch  Test Acc   Test Loss   LR        Time
  1    12.15%     2.2959      0.0600    252s
  5    93.94%     0.1911      0.0585    271s
 10    96.34%     0.1160      0.0528    271s
 20    97.34%     0.0861      0.0324    271s
 30    97.71%     0.0777      0.0105    282s
 38    97.96%     0.0761      0.0008    277s
 40    97.90%     0.0756      0.0001    279s
```

**Peak accuracy: 97.96%** (epoch 38)

### Comparison

| | Batch=32 | Batch=256 |
|---|---|---|
| Peak accuracy | **98.10%** | 97.96% |
| Updates/epoch | 1875 | 234 |
| Time/epoch | ~272s | ~273s |
| Convergence | 97%+ by epoch 6 | 97%+ by epoch 15 |

The smaller batch size yields slightly higher accuracy due to 8x more weight updates per epoch. Both converge to the same ~98% plateau. **Throughput: ~220 samples/s** on 32 threads.

## Analysis

### What 98.1% means

- A linear classifier on raw MNIST pixels achieves ~92%.
- A 2-layer MLP achieves ~98%.
- Standard 2D CNNs achieve 99.0-99.5%.

HypercubeCNN at 98.1% matches a well-tuned MLP -- which is the right comparison, because **both operate without spatial inductive bias**. The ~1% gap to spatial CNNs is the cost of not encoding 2D locality. This is expected and intentional -- the architecture is not designed for spatial data.

### Optimization history

| Configuration | Accuracy |
|---------------|----------|
| Shell+NN masks (K=2*DIM-2), scale=0.1, lr=0.16 | 94.8% |
| NN-only (K=DIM), Xavier init, lr=0.04 | 96.6% |
| Wider (32->64->128->128), 4th stage, L2 decay, cosine LR | **98.1%** |

Key changes that mattered:
1. **Removing shell masks** (+1.8% accuracy, 1.58x speedup) -- nearest-neighbor-only convolution outperformed full masks.
2. **Xavier/Glorot initialization** -- eliminated dead-network failures.
3. **Wider channels + 4th stage** -- ~35K to ~200K parameters.
4. **L2 weight decay + cosine LR** -- regularization and smooth convergence.

### What would push higher

Without spatial inductive bias, reaching 99%+ likely requires batch normalization, Adam optimizer, or data augmentation. BN and Adam are now fully supported (see [docs/CPP_SDK.md](../docs/CPP_SDK.md)); the runs above used SGD-only. A fresh sweep with BN + Adam would likely close some of the gap.

## Significance

98.1% on MNIST without spatial inductive bias demonstrates that hypercube convolution learns non-trivial image features from Hamming-distance relationships alone. These results validate the training pipeline for deployment on **native hypercube data** (molecular fingerprints, Boolean functions, reservoir state) where the Hamming-distance inductive bias is a structural advantage rather than a handicap.
