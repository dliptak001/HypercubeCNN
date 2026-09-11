# Spatial preprocess: augmentation and embed (P ≤ N)

Optional **SDK modules** for mapping 2D single-channel images onto HypercubeCNN
inputs. They ship in `HypercubeCNNCore` and are installed with the public
headers, but they are **not** part of the conv/pool graph and do **not** depend
on each other beyond a recommended pipeline order.

| Module | Header | Role | Knows DIM? |
|--------|--------|------|------------|
| Spatial aug | `HCNNSpatialAug.h` | Stochastic 2D geometry / noise on any **H×W** grid | No |
| Spatial embed | `HCNNSpatialEmbed.h` | Layout a 2D image into length **N = 2^dim** with **pattern length P ≤ N** | Yes (`dim`) |

These modules exist because network capacity is always **N = 2^dim** (hypercube
topology). Images and other non–power-of-two tensors are a **host packing**
problem: you (or these helpers) map them into length N; the conv graph never
sees “H×W” or “length 784.” Native cube data already at length ≤ N can ignore
both modules and use `HCNN::Embed` / `TrainEpoch` directly — or use any other
custom pack of your own.

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
- If your raw **H×W** product is greater than **N**, **PadLow** / **PadLowCenter**
  throw; use **ResizeToFit** or **DualPlaneResize**, or increase `dim`.

| dim | N | Max square S (`ResizeToFit`) | Max dual S (`DualPlaneResize`) |
|-----|---|------------------------------|--------------------------------|
| 9 | 512 | 22 (S² = 484) | **16** (2·S² = 512) |
| 10 | 1024 | 32 | 22 |
| 11 | 2048 | 45 | **32** (2·S² = 2048) |
| 12 | 4096 | 64 | 45 |

Auto side: `plane_side = 0` uses `floor(sqrt(N))` or `floor(sqrt(N/2))`.

---

## Choosing a mode

| Prefer | When |
|--------|------|
| **`PadLow`** | H×W already ≤ N and you want exact pixels only (no second view, no resize). |
| **`PadLowCenter`** | H×W ≤ N and you want the **full native image** plus a **center crop** packed into leftover vertices (good when rem is meaningful, e.g. 28×28 into N=1024). |
| **`ResizeToFit`** | H×W may exceed N, or you only need one square view and accept aspect distortion. |
| **`DualPlaneResize`** | You want intensity **and** edge structure on a single channel (ink ‖ \|grad\|); classic full fill at dim 9 / 11. |

In-tree **MNISTTrain** uses **DualPlaneResize @ dim=11** (engineered ~99% recipe).
**PadLowCenter @ dim=10** is a first-class alternative when you want native 28×28
plus a center crop without bilinear resize of the primary view.

Default `HCNNSpatialEmbedConfig::mode` / Python `SpatialEmbedder` mode is **`PadLow`**.

---

## Embed modes

### `PadLow`

```text
out[0 .. H*W)  = image (row-major)
out[H*W .. N)  = pad_value
```

- No resize. Requires **H×W ≤ N** (product of height and width).
- Use when the image already fits (small patches, downsampled offline).
- Formerly named **`RowMajorPad`** (renamed; **no** enum alias — see
  [ChangeLog.md](../ChangeLog.md)).

### `PadLowCenter`

```text
out[0 .. H*W)                    = image (row-major)
out[H*W .. H*W + crop_h*crop_w)  = centered crop (row-major)
out[H*W + crop_h*crop_w .. N)    = pad_value
```

- No resize of the primary view — keeps full native resolution in the low
  addresses.
- Requires **H×W ≤ N**. Remaining budget `R = N − H×W` is filled with the
  largest near-square center crop (area ≤ R, fits in H×W), floor-centered.
  Tie-break: min |h−w|, then prefer wider, then smaller h.
- MNIST **28×28 @ dim=10** (N=1024): crop **15×16 @ (6,6)**, full occupancy
  (`pattern_length = 1024`).
- Plan exposes `crop_h`, `crop_w`, `crop_row0`, `crop_col0` (all zero for other
  modes).

**Edge cases:**

- **R = 0** (H×W = N): crop is empty (`crop_h = crop_w = 0`). Layout matches
  **PadLow** (full image only).
- **R ≥ H×W**: the crop can be the **entire** image — the tail is a second
  full copy under the same max-area / near-square rule (common when N is large
  relative to the image).
- **Non-square H×W**: same algorithm; origin is still floor-centered on the
  chosen crop size.

### `ResizeToFit`

```text
S = plane_side or floor(sqrt(N))
out[0 .. S*S)  = bilinear resize of image to S×S
out[S*S .. N)  = pad_value
```

- Always succeeds for N ≥ 1 (embed `dim` is validated in [1, 30]; networks
  typically require dim ≥ 3 to train).
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
- Gradient of a blank/constant plane is filled with `pad_value`. For a
  *truly* blank resized plane, `pad_value` must match the constant ink (or
  OOB samples invent edges at the border).
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
- Prefer **shear A/B first**, then enable elastic (MNIST recipe defaults elastic off).

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

// 2) Embed into N = 2^dim  (DualPlane example — MNIST-style dim=11)
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

**PadLowCenter** (native full image + center crop; dim=10 MNIST packing):

```cpp
hcnn::HCNNSpatialEmbedConfig ecfg;
ecfg.dim = 10;
ecfg.mode = hcnn::HCNNSpatialEmbedMode::PadLowCenter;
ecfg.pad_value = -1.f;
hcnn::HCNNSpatialEmbedder emb(ecfg);
auto plan = emb.plan(28, 28);
// plan.crop_h/w = 15/16, plan.crop_row0/col0 = 6/6, plan.pattern_length = 1024
auto packed = hcnn::pack_spatial(emb, img28, 28, 28);
```

### Planning without embedding

```cpp
auto plan = emb.plan(height, width);
// Always useful:
//   plan.N, plan.pattern_length, plan.mode, plan.plane_side
// PadLowCenter also fills:
//   plan.crop_h, plan.crop_w, plan.crop_row0, plan.crop_col0
// (those four stay 0 for PadLow / ResizeToFit / DualPlaneResize)
```

### Batch

```cpp
aug.apply_batch(in, tmp, batch, H, W, rng);   // stride H*W
emb.embed_batch(tmp, batch, H, W, out);       // stride N; batch==0 is a no-op
```

---

## Design boundaries

| In scope | Out of scope (v1) |
|----------|-------------------|
| Single-channel 2D → length N | Multi-channel `c_in > 1` packs |
| Pad / pad+center crop / square resize / dual ink + \|grad\| | Locality-aware vertex scatter |
| Works for any `dim` in [1, 30] | Tying aug to Embed inside `HCNN` |
| Deterministic embed; RNG only in aug | Dataset I/O (IDX loaders stay examples) |

---

## Tests

`CoreSmokeTest` covers:

- Aug identity, determinism, border, noise, batch, error paths  
- Embed capacity, PadLow, PadLowCenter, ResizeToFit, DualPlaneResize  
- Reject H×W product > N for PadLow / PadLowCenter  
- PadLowCenter 28×28 @ dim=10 → 15×16 center, full occupancy  
- PadLowCenter non-square H×W, rem=0 (empty crop), batch `pad_value`, empty batch  
- Dual-plane full occupancy; blank plane → |grad| filled with `pad_value`  
- Fitting `plane_side` override  

Python: `python/tests/test_basic.py` (`TestSpatial`) and
`examples/python/spatial_embed_smoke.py` (PadLow, PadLowCenter, DualPlane).

---

## Related

| Doc / header | Content |
|--------------|---------|
| `HCNNSpatialAug.h` | Aug config + augmenter |
| `HCNNSpatialEmbed.h` | Embed config, modes, plan, embedder |
| `docs/CPP_SDK.md` | Public SDK; spatial is §9 (modes table + pad contract) |
| `docs/Python_SDK.md` | Python spatial section (modes + `SpatialEmbedPlan`) |
| `docs/internals.md` | Hypercube conv (after embed) |
| `examples/mnist_train.md` | Full image training recipe (DualPlane) |
