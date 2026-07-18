# Spatial preprocess: augmentation and embed (P ≤ N)

Optional **core** helpers for mapping 2D single-channel images onto HypercubeCNN
inputs. They are **not** part of the conv/pool graph and do **not** depend on
each other beyond a recommended pipeline order.

| Module | Header | Role | Knows DIM? |
|--------|--------|------|------------|
| Spatial aug | `HCNNSpatialAug.h` | Stochastic 2D geometry / noise on \(H\times W\) | No |
| Spatial embed | `HCNNSpatialEmbed.h` | Deterministic layout of a 2D image into length \(N=2^{\mathrm{dim}}\) with **pattern length \(P \le N\)** | Yes (`dim`) |

Native hypercube data (already length \(\le N\)) can ignore both modules and
use `HCNN::Embed` / `TrainEpoch` directly.

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
embed (resize/pad) into the cube. That keeps geometric aug rich when \(N\) is
small.

---

## Capacity: \(P \le N\)

- Hypercube input capacity per channel: \(N = 2^{\mathrm{dim}}\).
- Embed always writes a **full** buffer of length \(N\).
- Occupied pattern length \(P\) (before pad) always satisfies \(P \le N\).
- If your raw \(H\cdot W > N\), **RowMajorPad** throws; use **ResizeToFit** or
  **DualPlaneResize**, or increase `dim`.

| dim | N | Max square \(S\) (`ResizeToFit`) | Max dual \(S\) (`DualPlaneResize`) |
|-----|---|----------------------------------|-------------------------------------|
| 9 | 512 | 22 (\(S^2=484\)) | **16** (\(2S^2=512\)) |
| 10 | 1024 | 32 | 22 |
| 11 | 2048 | 45 | **32** (\(2S^2=2048\)) |
| 12 | 4096 | 64 | 45 |

Auto side: `plane_side = 0` uses `floor(sqrt(N))` or `floor(sqrt(N/2))`.

---

## Embed modes

### `RowMajorPad`

```text
out[0 .. H*W)  = image (row-major)
out[H*W .. N)  = pad_value
```

- No resize. Requires \(H\cdot W \le N\).
- Use when the image already fits (small patches, downsampled offline).

### `ResizeToFit`

```text
S = plane_side or floor(sqrt(N))
out[0 .. S*S)  = bilinear resize of image to S×S
out[S*S .. N)  = pad_value
```

- Always succeeds for \(N \ge 1\).
- Single view; unused vertices if \(S^2 < N\).

### `DualPlaneResize`

```text
S = plane_side or floor(sqrt(N/2))
out[0 .. S*S)         = bilinear resize (ink)
out[S*S .. 2*S*S)     = |∇| of ink plane, per-image max-norm → ~[-1, 1]
out[2*S*S .. N)       = pad_value
```

- Multi-view occupancy without multi-channel inputs.
- When \(N\) is a multiple of 2 and \(S=\lfloor\sqrt{N/2}\rfloor\), often
  \(2S^2 = N\) (full fill), e.g. dim 9 → 16×16‖|∇|, dim 11 → 32×32‖|∇|.
- Gradient of a blank/constant plane is filled with `pad_value`.

**Layout is row-major blocks, not locality-aware Hamming packing.** For
Cartesian Gray / Hilbert maps see `examples/mnist_locality_aware_packing.md`
(design memo; not this API).

---

## Augmentation (recap)

See `HCNNSpatialAug.h`:

- Config: rotate ±deg, scale range, integer shift, Gaussian noise, border, clip.
- One inverse bilinear warp for geometry; noise after.
- Any \(H\times W\); no `dim` field.
- Geometric path requires `in != out`.

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
const int N = emb.capacity();           // 2048
auto plan = emb.plan(H, W);             // plane_side=32, pattern_length=2048

std::vector<float> work(H * W), packed(N);
std::mt19937 rng(seed);
aug.apply(src28, work.data(), H, W, rng);
emb.embed(work.data(), H, W, packed.data());

// 3) Network (input_length = N)
hcnn::HCNN net(ecfg.dim, /*num_outputs=*/10, /*input_channels=*/1);
// ... AddConv, train with packed as input_length N ...
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
| Single-channel 2D → length \(N\) | Multi-channel `c_in > 1` packs |
| Pad / square resize / dual ink+\|∇\| | Locality-aware vertex scatter |
| Works for any `dim` in [1, 30] | Tying aug to Embed inside `HCNN` |
| Deterministic embed; RNG only in aug | Dataset I/O (IDX loaders stay examples) |

---

## Tests

`CoreSmokeTest` covers:

- Aug identity, determinism, border, noise, batch, error paths  
- Embed capacity, RowMajorPad, ResizeToFit, DualPlaneResize  
- Reject \(H\cdot W > N\) for RowMajorPad  
- Dual-plane full occupancy for classic powers of two  

---

## Related

| Doc / header | Content |
|--------------|---------|
| `HCNNSpatialAug.h` | Aug config + augmenter |
| `HCNNSpatialEmbed.h` | Embed config, modes, plan, embedder |
| `docs/CPP_SDK.md` | Install layout + pointers |
| `docs/architecture.md` | Hypercube conv (after embed) |
| `examples/mnist_locality_aware_packing.md` | Future locality maps (design) |
