# Spatial preprocess: augmentation and embed (P ≤ N)

Optional **core** helpers for mapping 2D single-channel images onto HypercubeCNN
inputs. They are **not** part of the conv/pool graph and do **not** depend on
each other beyond a recommended pipeline order.

| Module | Header | Role | Knows DIM? |
|--------|--------|------|------------|
| Spatial aug | `HCNNSpatialAug.h` | Stochastic 2D geometry / noise on any **H×W** grid | No |
| Spatial embed | `HCNNSpatialEmbed.h` | Layout a 2D image into length **N = 2^dim** with **pattern length P ≤ N** | Yes (`dim`) |

Native hypercube data (already length ≤ N) can ignore both modules and use
`HCNN::Embed` / `TrainEpoch` directly.

**See also:** end-to-end image recipe in [`examples/mnist_train.md`](../examples/mnist_train.md)
(DualPlane embed, train aug, `input_length = N`).

---

## Pipeline

```text
  H×W row-major image
        │
        │  optional: HCNNSpatialAugmenter  (any H,W; DIM-agnostic)
        ▼
  H×W (possibly warped)
        │
        │  HCNNSpatialEmbedder  (dim → N = 2^dim; mode chooses layout)
        ▼
  float[N]   (one input channel, vertex-major)
        │
        │  HCNN::TrainEpoch / ForwardBatch  (input_length = N)
        ▼
  network
```

**Rule of thumb:** augment at the **native** resolution you care about, then
embed (resize/pad) into the cube. That keeps geometric aug rich when N is
small.

### Caller contracts (important)

1. **Prefer full-capacity typed inputs** (`HCNNInputView` / `HCNNInputBatch` in
   `HCNNInput.h`). After spatial embed, use:
   - `pack_spatial` / `pack_spatial_batch`, or
   - `HCNNFlatDataset::input_view()` when `input_length == N`,  
   then `net.TrainEpoch(view, labels, batch, params)`.

2. **Train / infer with `input_length = emb.capacity()` (= N)** if you still use
   raw pointers. Spatial embed always fills a full length-N buffer (including pad).

3. **Do not pass `pattern_length < N` into `HCNN::Embed` / raw Train* if you used a
   non-zero `pad_value`.**  
   Network `Embed` always zero-pads the tail to capacity. That **overwrites**
   spatial-embed padding (e.g. -1) with **0**. Typed overloads reject capacity
   mismatch; raw APIs still allow short length for intentional zero-pad.

4. **Intentional short + zero pad** (native cube data): use
   `HCNNInputBatch::from_short_zero_pad` so the pad policy is explicit.

5. **Aug then embed** (not embed then aug on the packed vector). Aug is 2D only.

6. **DualPlane / digit-like data:** set `pad_value` to background (e.g. **-1**),
   not the default 0 — bilinear OOB and unused vertices use `pad_value`.

7. **`input_channels = 1`** for this helper. Multi-channel packing is custom.

---

## Capacity: P ≤ N

Here **P ≤ N** means the occupied pattern length **P** is at most the hypercube
capacity **N** (less-than-or-equal).

- Hypercube input capacity per channel: **N = 2^dim**.
- Embed always writes a **full** buffer of length **N**.
- Occupied pattern length **P** (before pad) always satisfies **P ≤ N**.
- If your raw **H×W** product is greater than **N**, **RowMajorPad** throws;
  use **ResizeToFit** or **DualPlaneResize**, or increase `dim`.

| dim | N | Max square S (`ResizeToFit`) | Max dual S (`DualPlaneResize`) |
|-----|---|------------------------------|--------------------------------|
| 9 | 512 | 22 (S² = 484) | **16** (2·S² = 512) |
| 10 | 1024 | 32 | 22 |
| 11 | 2048 | 45 | **32** (2·S² = 2048) |
| 12 | 4096 | 64 | 45 |

Auto side: `plane_side = 0` uses `floor(sqrt(N))` or `floor(sqrt(N/2))`.

---

## Embed modes

### `RowMajorPad`

```text
out[0 .. H*W)  = image (row-major)
out[H*W .. N)  = pad_value
```

- No resize. Requires **H×W ≤ N** (product of height and width).
- Use when the image already fits (small patches, downsampled offline).

### `ResizeToFit`

```text
S = plane_side or floor(sqrt(N))
out[0 .. S*S)  = bilinear resize of image to S×S
out[S*S .. N)  = pad_value
```

- Always succeeds for N ≥ 1.
- Single view; unused vertices if S² < N.
- **Always square:** non-square H×W is distorted to S×S.
- Bilinear OOB uses `pad_value`.

### `DualPlaneResize`

```text
S = plane_side or floor(sqrt(N/2))
out[0 .. S*S)         = bilinear resize (ink)
out[S*S .. 2*S*S)     = |grad| of ink plane, per-image max-norm → ~[-1, 1]
out[2*S*S .. N)       = pad_value
```

- Multi-view occupancy without multi-channel inputs.
- When N is even and S = floor(sqrt(N/2)), often **2·S² = N** (full fill),
  e.g. dim 9 → 16×16 ‖ |grad|, dim 11 → 32×32 ‖ |grad|.
- Gradient of a blank/constant plane is filled with `pad_value`.
- Ink is **not** range-clipped; only |grad| is max-normalized to ~[-1, 1].
- Bilinear OOB uses `pad_value` (use -1 for MNIST-like backgrounds).

**Layout is row-major blocks, not locality-aware Hamming packing.** Bit-flip
neighbors on the cube are not automatic 2D pixel adjacency under this embed.

---

## Augmentation (recap)

See `HCNNSpatialAug.h`:

- **Affine (one inverse bilinear warp):** rotate ±deg, scale range, integer
  shift, shear_x / shear_y (about center). Order: scale → shear → rotate → shift.
  Require `|shear_x_max| * |shear_y_max| < 0.95` (invertible shear).
- **Elastic (optional, after affine):** Simard-style smooth random displacement;
  `elastic_alpha` = max |component| in pixels after field normalize;
  `elastic_sigma` in `[0.25, 32]` when alpha > 0.
  Cost is **O(H·W·⌈3σ⌉)** per field (two fields) — usually dominates aug time.
- **Noise:** Gaussian after geometry; clip to `[value_min, value_max]`.
- Any **H×W**; no `dim` field.
- Affine or elastic requires `in != out`. Elastic uses thread_local scratch
  (grows to max size seen on the thread).
- Prefer **shear A/B first**, then enable elastic (MNIST demo defaults elastic off).

---

## API sketch

```cpp
#include "HCNNSpatialAug.h"
#include "HCNNSpatialEmbed.h"
#include "HCNN.h"

// 1) Optional aug (DIM-free)
hcnn::HCNNSpatialAugConfig acfg;
acfg.rot_deg_max = 12.f;
acfg.scale_min = 0.9f;
acfg.scale_max = 1.1f;
acfg.shift_max = 2;
acfg.shear_x_max = 0.15f;   // next-level: shear first
// acfg.elastic_alpha = 1.f; // optional after shear A/B
// acfg.elastic_sigma = 5.f;
acfg.noise_sigma = 0.03f;
acfg.border_value = -1.f;
hcnn::HCNNSpatialAugmenter aug(acfg);

// 2) Embed into N = 2^dim
hcnn::HCNNSpatialEmbedConfig ecfg;
ecfg.dim = 11;
ecfg.mode = hcnn::HCNNSpatialEmbedMode::DualPlaneResize;
ecfg.pad_value = -1.f;
hcnn::HCNNSpatialEmbedder emb(ecfg);

const int H = 28, W = 28;
auto plan = emb.plan(H, W);             // plane_side=32, pattern_length=2048

std::vector<float> work(H * W);
std::mt19937 rng(seed);
aug.apply(src28, work.data(), H, W, rng);

// 3) Full-capacity pack (pad_value preserved on unused verts)
auto packed = hcnn::pack_spatial(emb, work.data(), H, W);
// packed.capacity() == emb.capacity() (== N)

// 4) Network — typed path (recommended)
hcnn::HCNN net(ecfg.dim, /*num_outputs=*/10, /*input_channels=*/1);
// net.TrainStep(packed.view(), label, params);
// or: net.TrainEpoch(ds.input_view(), labels, batch, params);
```

### Planning without embedding

```cpp
auto plan = emb.plan(height, width);
// plan.N, plan.plane_side, plan.pattern_length, plan.mode
```

### Batch

```cpp
aug.apply_batch(in, tmp, batch, H, W, rng);   // stride H*W
emb.embed_batch(tmp, batch, H, W, out);       // stride N
```

---

## Design boundaries

| In scope | Out of scope (v1) |
|----------|-------------------|
| Single-channel 2D → length N | Multi-channel `c_in > 1` packs |
| Pad / square resize / dual ink + \|grad\| | Locality-aware vertex scatter |
| Works for any `dim` in [1, 30] | Tying aug to Embed inside `HCNN` |
| Deterministic embed; RNG only in aug | Dataset I/O (IDX loaders stay examples) |

---

## Tests

`CoreSmokeTest` covers:

- Aug identity, determinism, border, noise, batch, error paths  
- Embed capacity, RowMajorPad, ResizeToFit, DualPlaneResize  
- Reject H×W product > N for RowMajorPad  
- Dual-plane full occupancy for classic powers of two  

---

## Related

| Doc / header | Content |
|--------------|---------|
| `HCNNSpatialAug.h` | Aug config + augmenter |
| `HCNNSpatialEmbed.h` | Embed config, modes, plan, embedder |
| `docs/CPP_SDK.md` | SDK overview (§8: short pointer + pad contract) |
| `docs/internals.md` | Hypercube conv (after embed) |
| `examples/mnist_train.md` | Full image training demo |
