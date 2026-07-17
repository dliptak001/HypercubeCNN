# MNIST Training -- Handwritten Digit Classification

Demonstrates end-to-end training and evaluation of a HypercubeCNN on the MNIST handwritten digit dataset (60K train, 10K test, 10 classes).

## What this example shows

- Loading real MNIST data from IDX binary files
- Constructing a 2-stage conv+pool architecture with FLATTEN readout (DIM=11)
- Training with mini-batch Adam, cosine LR annealing, and weight decay
- Dual checkpoints: best test loss and best test accuracy (via `GetWeights` / `SetWeights`)
- Parallel batch inference for evaluation

## How MNIST maps onto the hypercube

MNIST images are 28×28 = 784 grayscale pixels. The network uses **DIM=11**, giving N = 2^11 = **2048** vertices. Pixels are normalized to [-1.0, 1.0] and assigned to vertices 0–783 via Direct Linear Assignment. Vertices 784–2047 are zero-padded.

No spatial locality is encoded -- the 2D pixel grid is flattened and mapped onto hypercube vertices in index order. The network must learn all useful relationships from the hypercube topology alone.

## Architecture

```
Input: 784 pixels -> 2048 vertices (DIM=11)
  |
Conv1: 1 -> 16 channels, K=11, ReLU, bias
Pool1: MAX (antipodal), DIM 11->10, N 2048->1024
  |
Conv2: 16 -> 16 channels, K=10, ReLU, bias
  |
Readout: FLATTEN -> linear 16384->10 -> logits
```

Total parameters: **166,618** (192 conv1 + 2,576 conv2 + 163,850 readout).

The readout dominates the parameter count (~98%). FLATTEN treats every (channel, vertex) activation as an independent feature -- the linear layer learns per-vertex weights, which is well-suited to classification where vertex identity carries information. Larger N (vs DIM=10) mainly grows this head: more (channel, vertex) features after pooling, not a deeper hierarchy.

## Training configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Optimizer | Adam | Decoupled weight decay (AdamW), default betas (0.9, 0.999) |
| Learning rate | `lr_max = 0.001` | Peak LR on the first epoch |
| LR schedule | Cosine anneal `lr_max → 0.1·lr_max` | Floor is **1e-4**. Progress uses `epoch/(epochs-1)` so the last epoch hits `lr_min` exactly. No warmup or restarts. |
| Batch size | 256 | Parallel across threads via `TrainBatch` (dispatched by `TrainEpoch`) |
| Weight decay | 1e-3 | Applied to kernels and readout weights (not biases). Stronger WD suits the readout-heavy model |
| Epochs | 40 | Late epochs fine-tune under a small LR; dual checkpoints still pick best-loss vs best-acc |
| Shuffle | per-epoch | `TrainEpoch(..., shuffle_seed = epoch + 1)` |
| Checkpoints | dual | Best test **loss** and best test **acc** via `GetWeights` / `SetWeights`; network left on best-acc weights after training |

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
cmake --build cmake-build-release --target MNISTTrain
./cmake-build-release/MNISTTrain
```

## Results

60K train / 10K test, **DIM=11**, Adam, batch=256, weight decay **1e-3**, cosine LR **0.001 → 1e-4** (10% floor), 40 epochs.

```
Epoch  Test Acc    Test Loss   LR        Time/epoch
  1    94.49%      0.1885      0.00100   ~16s
  4    97.26%      0.0920      0.00099   ~16s
  6    97.74%      0.0714      0.00096   ~17s
  9    98.00%      0.0676      0.00091   ~16s   ← best loss
 14    98.21%      0.0693      0.00078   ~16s
 18    98.25%      0.0772      0.00064   ~16s
 24    98.26%      0.0860      0.00042   ~17s
 37    98.27%      0.0939      0.00011   ~17s   ← best acc
 40    98.24%      0.0951      0.00010   ~18s
```

| Checkpoint | Epoch | Test loss | Test acc |
|------------|-------|-----------|----------|
| **Best loss** | 9 | **0.0676** | 98.00% |
| **Best acc** | 37 | 0.0939 | **98.27%** |

**Throughput: ~3,400–3,800 samples/s** on 32 threads (~16–17 s/epoch).

After the loss minimum, CE rises modestly while accuracy creeps then plateaus near 98.26%. Dual checkpoints keep both the well-calibrated early model and the peak top-1 model; the run ends with best-acc weights restored.

## Analysis

### What ~98.3% means

- A linear classifier on raw MNIST pixels achieves ~92%.
- A 2-layer MLP achieves ~98%.
- Standard 2D CNNs achieve 99.0–99.5%.

HypercubeCNN at **98.27%** sits with a well-tuned MLP -- the right comparison, because **both operate without spatial inductive bias**. The remaining gap to spatial CNNs is the cost of not encoding 2D locality. That is expected and intentional -- the architecture is not designed for spatial image data.

### Architecture choices

The shallow stack (2 conv layers, 1 pool, FLATTEN readout) was chosen over a deeper multi-stage stack because:

1. **FLATTEN makes depth less critical.** The readout sees every (channel, vertex) activation directly, so it can learn per-vertex discriminative features without a deep feature hierarchy.
2. **Practical wall-clock.** ~16–17 s/epoch at DIM=11 on 32 threads -- still easy to iterate; a deep 4-stage stack is much slower for similar peak accuracy in earlier experiments.
3. **DIM=11 vs DIM=10.** Doubling N (1024→2048) roughly doubles readout capacity and improves both best loss (~0.078→0.068 under the same train recipe) and peak accuracy (~98.0%→98.27%), at about 2× epoch time. Further DIM growth scales the linear head aggressively and is usually the wrong next lever compared to regularization, width, or a locality-aware embed.

## Significance

**98.27%** on MNIST without spatial inductive bias shows that hypercube convolution learns non-trivial digit structure from Hamming-distance relationships alone. These results validate the training pipeline for deployment on **native hypercube data** (molecular fingerprints, Boolean functions, reservoir state) where the Hamming-distance inductive bias is a structural advantage rather than a handicap.
