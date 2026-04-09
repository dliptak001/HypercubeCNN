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

The `HCNNDataset` struct holds a vector of `Sample` (each with `std::vector<float> input` and `int target_class`). The example pre-builds a `DatasetView` of raw pointer arrays once per dataset and feeds it to `HCNN::TrainEpoch` (which handles shuffling and batching).

The shipped example loads a 20K / 2K subset for fast iteration. Bump both calls to `60000` / `10000` to train on the full dataset (~98% accuracy after 20–40 epochs — see [docs/mnist.md](../docs/mnist.md)).

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

```
Loading MNIST from .../data...
Train: 60000 samples, Test: 10000 samples
Threads: 32

=== HCNN (lr=0.06, batch=256, wd=0.0001) ===
Initial test: loss=2.302... acc=980/10000 (9.8%)
Epoch 1/40: loss=0.35... acc=8900/10000 (89.0%)
  (lr=0.059..., 3.2s, 18750 samples/s)
...
Epoch 40/40: loss=0.06... acc=9800/10000 (98.0%)
  (lr=0.00001, 3.1s, 19354 samples/s)
```

Exact numbers vary by platform and thread count. Final accuracy is typically ~98%.
