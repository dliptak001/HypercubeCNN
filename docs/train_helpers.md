# Training helpers

Optional **core** utilities for classification training loops. They are **not**
part of the conv/pool graph and do **not** change `HCNN` behavior. Include
`HCNNTrainHelpers.h` when an example or app wants a thin loop instead of
re-implementing CE / cosine / dual checkpoints.

| Piece | API | Role |
|-------|-----|------|
| Metrics | `argmax`, `softmax_cross_entropy`, `evaluate_classification`, `HCNNClassEval` | CE + accuracy over a flat batch |
| Flat dataset | `HCNNFlatDataset` | Contiguous inputs + int labels for `TrainEpoch` / `ForwardBatch` |
| Cosine LR | `cosine_lr(lr_max, lr_min, epoch, num_epochs)` | Anneal from max → min over epochs |
| Dual checkpoint | `HCNNDualCheckpoint` | Best test loss and best test accuracy weight blobs |

Native hypercube workloads that already own their loop can ignore this header.

---

## Metrics

```cpp
#include "HCNNTrainHelpers.h"

hcnn::HCNNClassEval r = hcnn::evaluate_classification(
    net, flat_inputs, input_length, targets, count);
// r.loss (mean CE), r.accuracy (percent), r.correct, r.count
```

Or with a flat dataset:

```cpp
hcnn::HCNNFlatDataset ds;
ds.reset(n, input_length);
// fill ds.inputs / ds.targets ...
auto r = hcnn::evaluate_classification(net, ds);
```

`softmax_cross_entropy` and `argmax` are available for custom eval loops.

---

## Cosine LR

`HCNN` does **not** own a learning-rate schedule. Call sites pass `lr` into
each `TrainEpoch` / `TrainBatch` / `TrainStep`.

```cpp
// epoch is 0-based; last epoch reaches lr_min when num_epochs > 1
float lr = hcnn::cosine_lr(/*lr_max=*/1e-3f, /*lr_min=*/1e-4f, epoch, /*num_epochs=*/60);
net.TrainEpoch(..., lr, ...);
```

Formula (same as the documented MNIST schedule):

```text
progress = epoch / (num_epochs - 1)     # clamped
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

Typical floor: `lr_min = 0.1 * lr_max`.

---

## Dual checkpoints

Tracks two independent `GetWeights()` snapshots:

- **Best loss** — lower CE wins; higher accuracy is the tie-break
- **Best accuracy** — higher percent wins; lower loss is the tie-break

```cpp
hcnn::HCNNDualCheckpoint ckpt;
for (int epoch = 0; epoch < epochs; ++epoch) {
    // train + evaluate ...
    auto upd = ckpt.observe(net, r.loss, r.accuracy, /*epoch_1based=*/epoch + 1);
    if (upd.new_best_loss) { /* log */ }
    if (upd.new_best_acc)  { /* log */ }
}
ckpt.restore_best_acc(net);
// or ckpt.restore_best_loss(net);
```

**Weights only.** `GetWeights` / `SetWeights` currently omit BN γ/β, and the
blob never includes optimizer state (SGD velocity / Adam moments / timestep).
Dual checkpoints are exact for **eval / export** on the no-BN stacks used in
the MNIST demo. Restoring a snapshot and continuing training reuses stale
optimizer moments unless you also reset them (e.g. `SetOptimizer`). BN nets
need extra care until γ/β land in the weight blob.

---

## Flat dataset

```cpp
hcnn::HCNNFlatDataset train;
train.reset(n, input_length);   // sizes inputs = n*len, targets = n
// train.sample_input(i) -> float* of length input_length
// train.targets[i] = class index
net.TrainEpoch(train.inputs.data(), train.input_length,
               train.targets.data(), train.count, batch_size, lr, ...);
```

Regression demos use float targets and keep their own flat layout; this helper
is classification-only.

---

## Relation to spatial preprocess

Typical image demo pipeline:

```text
H×W image
  -> optional HCNNSpatialAugmenter
  -> HCNNSpatialEmbedder  (length N)
  -> fill HCNNFlatDataset
  -> TrainEpoch + cosine_lr + evaluate_classification + HCNNDualCheckpoint
```

See [`spatial_preprocess.md`](spatial_preprocess.md) for aug/embed details.
`examples/mnist_train.cpp` uses Spatial* preprocess plus these train helpers
as the thin teaching loop.
