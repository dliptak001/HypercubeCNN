# MNIST Training -- Handwritten Digit Classification

Demonstrates end-to-end training and evaluation of a HypercubeCNN on the MNIST handwritten digit dataset (60K train, 10K test, 10 classes).

## What this example shows

- Loading real MNIST data from IDX binary files
- Constructing a 2-stage conv+pool architecture with FLATTEN readout
- Training with mini-batch Adam, cosine LR annealing, and weight decay
- Parallel batch inference for evaluation

## How MNIST maps onto the hypercube

MNIST images are 28x28 = 784 grayscale pixels. The network uses DIM=10, giving N = 2^10 = 1024 vertices. Pixels are normalized to [-1.0, 1.0] and assigned to vertices 0-783 via Direct Linear Assignment. Vertices 784-1023 are zero-padded.

No spatial locality is encoded -- the 2D pixel grid is flattened and mapped onto hypercube vertices in index order. The network must learn all useful relationships from the hypercube topology alone.

## Architecture

```
Input: 784 pixels -> 1024 vertices (DIM=10)
  |
Conv1: 1 -> 16 channels, K=10, ReLU, bias
Pool1: MAX (antipodal), DIM 10->9, N 1024->512
  |
Conv2: 16 -> 16 channels, K=9, ReLU, bias
  |
Readout: FLATTEN -> linear 8192->10 -> logits
```

Total parameters: ~84K (176 conv1 + 2,320 conv2 + 81,930 readout).

The readout dominates the parameter count. FLATTEN treats every (channel, vertex) activation as an independent feature -- the linear layer learns per-vertex weights, which is well-suited to classification where vertex identity carries information.

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | Adam | Decoupled weight decay (AdamW), default betas (0.9, 0.999) |
| Learning rate | 0.002 | Initial LR, decays via cosine annealing |
| LR schedule | Cosine annealing to 1e-3 | Smooth decay, no warmup or restarts |
| Batch size | 256 | Parallel across threads via `TrainBatch` (dispatched by `TrainEpoch`) |
| Weight decay | 5e-4 | Applied to kernels and readout weights (not biases) |
| Epochs | 40 | |
| Shuffle | per-epoch | `TrainEpoch(..., shuffle_seed = epoch + 1)` |

## Data loading

MNIST data is loaded from IDX binary files using `load_mnist()` from `dataloader/HCNNDataset.h`:

```cpp
auto train_data = load_mnist("data/train-images-idx3-ubyte",
                             "data/train-labels-idx1-ubyte", 60000);
auto test_data  = load_mnist("data/t10k-images-idx3-ubyte",
                             "data/t10k-labels-idx1-ubyte",  10000);
```

The `HCNNDataset` struct holds a vector of `Sample` (each with `std::vector<float> input` and `int target_class`). The example flattens these into a contiguous `FlatDataset` buffer (one `float[]` for all inputs, one `int[]` for all targets) and feeds it to `HCNN::TrainEpoch` (which handles shuffling and batching).

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

## Results

60K train / 10K test, Adam, batch=256, weight decay 5e-4, cosine LR 0.002 → 1e-3.

```
Epoch  Test Acc    Test Loss   LR        Time/epoch
  1    94.24%      0.1994      0.00200   ~6s
  4    97.41%      0.0894      0.00199   ~7s
  6    97.65%      0.0781      0.00196   ~7s
 12    97.62%      0.1002      0.00182   ~7s
 22    97.65%      0.1285      0.00146   ~7s
 27    98.02%      0.1290      0.00127   ~8s
 29    98.07%      0.1323      0.00121   ~8s
 40    98.05%      0.1391      0.00100   ~8s
```

**Peak accuracy: 98.07%** (epoch 29), **throughput: ~8,000-10,000 samples/s** on 32 threads.

## Analysis

### What 98% means

- A linear classifier on raw MNIST pixels achieves ~92%.
- A 2-layer MLP achieves ~98%.
- Standard 2D CNNs achieve 99.0-99.5%.

HypercubeCNN at 98% matches a well-tuned MLP -- which is the right comparison, because **both operate without spatial inductive bias**. The ~1% gap to spatial CNNs is the cost of not encoding 2D locality. This is expected and intentional -- the architecture is not designed for spatial data.

### Architecture choices

The current shallow architecture (2 conv layers, 1 pool, FLATTEN readout) was chosen over a deeper 4-stage stack because:

1. **FLATTEN readout makes depth less critical.** The readout sees every (channel, vertex) activation directly, so it can learn per-vertex discriminative features without needing deep feature hierarchies.
2. **Faster iteration.** ~7s/epoch vs. ~270s/epoch with the deep architecture, enabling rapid experimentation.
3. **Comparable accuracy.** The shallow net reaches 98% with Adam + full data -- matching the deep architecture's peak.

## Significance

98% on MNIST without spatial inductive bias demonstrates that hypercube convolution learns non-trivial image features from Hamming-distance relationships alone. These results validate the training pipeline for deployment on **native hypercube data** (molecular fingerprints, Boolean functions, reservoir state) where the Hamming-distance inductive bias is a structural advantage rather than a handicap.
