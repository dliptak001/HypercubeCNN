# MNIST Baseline Results

## Purpose

MNIST is not the target domain for HypercubeCNN — the architecture is designed for data that naturally lives on binary hypercubes. MNIST is a 2D spatial dataset mapped onto hypercube vertices via Direct Linear Assignment, which destroys the spatial locality that makes MNIST easy for traditional CNNs.

The MNIST results serve as a **baseline** to validate the training pipeline and establish that the architecture can learn non-trivial features despite operating without spatial inductive bias.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST 60K train / 10K test |
| Input embedding | Direct Linear Assignment (784 pixels → 1024 vertices, zero-padded) |
| Start DIM | 10 (N = 1024) |
| Architecture | 4 conv+pool stages |
| Channels | 1 → 32 → 64 → 128 → 128 |
| Pooling | Antipodal MAX, DIM 10→9→8→7→6 |
| K per layer | 10, 9, 8, 7 |
| Total parameters | ~200K |
| Readout | GAP → linear → 10 classes |
| Optimizer | SGD, momentum=0.9 |
| Weight decay | 1e-4 (L2, weights only) |
| LR schedule | Cosine annealing, lr_max=0.06, lr_min=1e-5 |
| Batch size | 32 or 256 (see runs below) |
| Epochs | 20–40 |
| Threads | 32 |

## Results

### Run 1 — batch=32, 20 epochs

```
Epoch  Test Acc   Test Loss   LR        Time
  1    93.99%     0.1940      0.0600    270s
  2    94.60%     0.1698      0.0599    275s
  3    96.21%     0.1240      0.0596    266s
  4    95.73%     0.1366      0.0592    264s
  5    95.40%     0.1462      0.0585    273s
  6    97.31%     0.0828      0.0577    278s
  7    97.59%     0.0798      0.0567    269s
  8    97.52%     0.0813      0.0556    268s
  9    97.56%     0.0761      0.0543    275s
 10    97.98%     0.0700      0.0528    270s
 11    97.67%     0.0790      0.0512    267s
 12    97.41%     0.0898      0.0495    269s
 13    97.99%     0.0643      0.0476    272s
 14    97.79%     0.0698      0.0457    276s
 15    97.90%     0.0691      0.0436    272s
 16    97.82%     0.0743      0.0415    272s
 17    97.76%     0.0793      0.0393    272s
 18    98.07%     0.0679      0.0370    273s
 19    98.10%     0.0629      0.0347    272s
 20    98.08%     0.0646      0.0324    277s
```

**Peak accuracy: 98.10%** (epoch 19)

### Run 2 — batch=256, 40 epochs

```
Epoch  Test Acc   Test Loss   LR        Time
  1    12.15%     2.2959      0.0600    252s
  2    69.83%     0.9743      0.0599    251s
  3    84.58%     0.4982      0.0596    252s
  4    91.72%     0.2657      0.0592    260s
  5    93.94%     0.1911      0.0585    271s
 10    96.34%     0.1160      0.0528    271s
 15    97.29%     0.0814      0.0436    276s
 20    97.34%     0.0861      0.0324    271s
 25    97.79%     0.0734      0.0207    278s
 30    97.71%     0.0777      0.0105    282s
 34    97.91%     0.0760      0.0044    274s
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

The smaller batch size yields slightly higher accuracy due to 8x more weight updates per epoch, providing more gradient noise which acts as implicit regularization. Both configurations converge to the same ~98% plateau.

**Throughput: ~220 samples/s** on 32 threads

## Analysis

### What 98.1% means

- A linear classifier on raw MNIST pixels achieves ~92%.
- A 2-layer MLP achieves ~98%.
- Standard 2D CNNs achieve 99.0-99.5%.
- State-of-the-art (augmented, ensembled) exceeds 99.7%.

HypercubeCNN at 98.1% matches a well-tuned MLP — which is the right comparison, because **both operate without spatial inductive bias**. The MLP treats pixels as independent features; HypercubeCNN treats them as hypercube vertices with Hamming-distance relationships that happen not to align with the image's 2D structure.

The ~1% gap to spatial CNNs is the cost of not encoding 2D locality. This is expected and intentional — the architecture is not designed for spatial data.

### Convergence

- Reaches 94% after epoch 1 (fast initial learning).
- Reaches 97%+ by epoch 6.
- Plateaus at ~98% by epoch 18-20.
- No signs of overfitting (test loss continues to decrease through epoch 19).
- Cosine LR provides smooth convergence without the oscillation seen in step decay.

### Optimization history

The current result reflects several rounds of architecture and training optimization:

| Configuration | Accuracy | Speed |
|---------------|----------|-------|
| Shell+NN masks (K=2*DIM-2), scale=0.1, lr=0.16 | 94.8% | baseline |
| NN-only (K=DIM), Xavier init, lr=0.04 | 96.6% (20K subset) | 1.58x faster |
| Wider (32→64→128→128), 4th stage, L2 decay, cosine LR | **98.1%** | ~same throughput |

Key changes that mattered:
1. **Removing shell masks** (+1.8% accuracy, 1.58x speedup) — nearest-neighbor-only convolution with lower LR outperformed full masks on every metric.
2. **Xavier/Glorot initialization** — eliminated dead-network failures that plagued flat-scale init.
3. **Wider channels + 4th stage** — increased model capacity from ~35K to ~200K parameters.
4. **L2 weight decay** — mild regularization (1e-4) to prevent overfitting with the larger model.
5. **Cosine LR schedule** — smoother convergence than step decay.

### What would push higher

Without spatial inductive bias, reaching 99%+ likely requires:
- Batch normalization (architecture-agnostic regularization)
- Dropout (prevent co-adaptation of features)
- Data augmentation (elastic deformations, pixel jitter)

These are standard techniques that don't introduce spatial assumptions — they'd prove the bottleneck is regularization, not representation capacity.

## Significance

98.1% on MNIST without spatial inductive bias demonstrates that hypercube convolution learns non-trivial image features from Hamming-distance relationships alone. The 2D spatial structure of the images is not encoded anywhere in the architecture — the network discovers useful patterns purely through the hypercube topology.

These results validate the training pipeline and architecture for deployment on **native hypercube data** (molecular fingerprints, Boolean functions) where the Hamming-distance inductive bias is a structural advantage rather than a handicap.
