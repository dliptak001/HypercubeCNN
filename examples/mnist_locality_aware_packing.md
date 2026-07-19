# Locality-Aware Packing for 2D Images on HypercubeCNN

Design memo: how to map 2D image data onto hypercube vertices so that
HypercubeCNN’s Hamming convolutions see **spatially meaningful** structure,
with reasonable assumptions and expectations. Grounded in the current MNIST
demo stack (`examples/mnist_train.cpp`) and core geometry (`docs/architecture.md`).

**Status:** design only — not yet implemented. Empirical MNIST numbers cited
below are for the **current row-major dense pack** (no locality-aware map).

---

## 1. What problem we are really solving

Hypercube convolution is not a free-form graph network. For each output
channel and vertex \(v\):

\[
\mathrm{out}[v]
  = b
  + \sum_{c,\,k}
      w_{c,k}\,\mathrm{in}\bigl[c,\; v \oplus 2^{k}\bigr]
\]

Consequences:

| Fact | Implication for packing |
|------|-------------------------|
| Edges = **single-bit flips** | “Neighbors” are fixed by vertex **addresses**, not by pixel adjacency unless we choose addresses carefully. |
| Weights are **shared by direction \(k\)** | Bit \(k\) is a **global axis** of the cube (same involution at every vertex). |
| Direction \(k\) is not “an edge type in image space” unless packing makes it so | Packing is the only place we define what each axis *means*. |
| Depth ≈ stacked bit hops | After \(L\) layers, information can travel roughly Hamming distance \(L\) (and more via multiple paths). |

**Packing** is the map

\[
\text{pixel }(y,x)\ \text{(and optional views)}
\;\longrightarrow\;
\text{vertex } v \in \{0,\ldots,2^{\mathrm{DIM}}-1\}.
\]

**Locality-aware packing** means:

> Arrange pixel → vertex so that **bit-flip directions become useful spatial
> (or multi-scale / multi-view) axes**, not a scramble of row-major order.

It does **not** mean “turn the hypercube into a perfect 2D grid.” That is
impossible in general. It means **align the image with the cube’s product
structure**.

### What success looks like

1. **Geometric:** 4-neighbors on the image grid often sit at **Hamming
   distance 1** on the cube (or on a designated bit field).
2. **Architectural:** shared weights \(w_k\) become interpretable (e.g. some
   \(k\) ≈ horizontal, some ≈ vertical, one \(k\) ≈ ink↔edge at same site).
3. **Empirical (optional):** better or faster accuracy under the same train
   recipe — *not guaranteed* on an already strong FLATTEN baseline.

---

## 2. Current MNIST pack (what it optimizes)

**Recipe (documented in `mnist_train.md`):**

- DIM = 11, \(N = 2048\), one input channel.
- 28×28 → 32×32 bilinear ink plane.
- Finite-difference \|∇\| on 32×32 → second plane.
- Layout:

| Vertices | Content |
|----------|---------|
| \(v \in [0,\,1024)\) | 32×32 ink, **row-major** |
| \(v \in [1024,\,2048)\) | 32×32 \|∇\|, **row-major** |

**Optimizes:**

1. Full occupancy (no zero pad of unused verts).
2. A second view (edges) without multi-channel input.
3. Implementation simplicity.

**Does not optimize:** Hamming locality. Under row-major linear indexing:

- Horizontal neighbors differ by 1 in the linear index — bit patterns are not
  systematically one clean spatial axis.
- Vertical neighbors differ by 32 — not “one bit = one row step.”
- The high bit that separates the two halves makes **every antipodal pair
  straddle ink-half and grad-half** (verified: 1024/1024 cross-half pairs at
  DIM=11). That is hostile to antipodal pooling and unrelated to spatial
  adjacency.

**Empirical context:** with **no antipodal pool**, geometric train aug
(rot ±12°, scale [0.9, 1.1], shift ±2, noise), 60 epochs, **3-conv** 16-wide
stack, best-acc **99.28%** on seed `398479293` with default shear_x (pre-shear
peak was 99.31%; best-loss CE ~0.021). So packing is **not** “fix a broken
model.” It is “give Hamming kernels something coherent to do,” so conv layers
are more than mild scrubbers in front of a large FLATTEN head.

**Reasonable accuracy expectation if packing is improved:** on the order of
**+0.05–0.2 pp** mean, **null** within seed noise, or a **slight regression**
if the head was exploiting row-major index structure. Treat as an experiment,
not a promised climb to 99.5%.

---

## 3. The right mathematical picture

### 3.1 The hypercube is a product of bits

\[
\{0,1\}^{\mathrm{DIM}}
\;\cong\;
\{0,1\}^{d_y}
\times
\{0,1\}^{d_x}
\times
\{0,1\}^{d_{\mathrm{extra}}}
\]

A **32×32** plane is exactly \(2^5 \times 2^5\). So **one plane needs 10 bits**
for a perfect Cartesian address:

- 5 bits → row  
- 5 bits → column  

Current demo: **DIM = 11** and **two** planes → the 11th bit is a natural
**plane / view** bit.

That match is structural, not folklore:

```text
v = [ plane bit ]  ||  [ y-field 5 bits ]  ||  [ x-field 5 bits ]
```

(or any fixed permutation of those fields).

| Bit flips in… | Rough meaning |
|---------------|----------------|
| \(x\)-field | change column (walk on \(x\)) |
| \(y\)-field | change row |
| plane bit | ink ↔ \|∇\| at the **same** \((y,x)\) |

The plane-bit edge is easy to underestimate: with Cartesian packing it is a
**same-location multi-view** mix, not a random long-range pair. A shared
direction weight \(w_k\) for that bit is exactly “how much to blend intensity
vs gradient **at this site**” — well matched to weight sharing by direction.

### 3.2 Binary vs Gray coordinates

If \(x\) is stored as plain binary, \(x\) and \(x+1\) can differ in **many**
bits (classic example: 15 → 16).

If \(x\) is **binary-reflected Gray code**:

- successive integers differ by **exactly one bit** inside the \(x\)-field.

**Default recommendation for axis-aligned locality:**

```text
v = (plane << 10) | (gray5(y) << 5) | gray5(x)
```

Then a unit step in \(x\) (resp. \(y\)) is Hamming distance 1 in the
appropriate field (border / Gray seam behavior should be documented and
tested).

**Why this fits HypercubeCNN better than a pure space-filling curve for a
first cut:** the kernel is **factorized by bit index \(k\)**, not “along a
curve order.” Labeling bits as \(x\)-bits and \(y\)-bits gives \(w_k\) a
stable spatial meaning.

### 3.3 Hilbert / Morton (role in the design space)

Map 2D → 1D with a locality-preserving curve, then Gray-code the 1D index
into 10 bits; use bit 10 as plane.

| | Cartesian Gray fields | Hilbert → Gray |
|--|----------------------|----------------|
| Average spatial clustering | strong on axes | strong on average |
| Bit \(k\) = axis? | **yes** (by construction) | **no** |
| Match to shared \(w_k\) | excellent | weaker / opaque |
| Implementation | tiny Gray tables + scatter | curve encoder + scatter |

**Recommendation:** Cartesian Gray = **Plan A**. Hilbert = **Plan B /
ablation**, not the default story for this architecture.

### 3.4 Fundamental limits (keep expectations honest)

1. **Degree mismatch.** Grid interior degree 4; hypercube degree = DIM
   (11 here). Extra directions *will* mix non-4-neighbor structure. Use them
   deliberately (plane, scale, hierarchy), do not pretend they vanish.

2. **No isometry.** You cannot preserve all Euclidean distances as Hamming
   distances. Optimize for **4-neighborhood + multi-scale / multi-view**, not
   all pairs.

3. **FLATTEN still owns a lot.** Even with perfect packing, the linear head
   can ignore locality. Packing helps when early layers learn better features;
   it does not force the model to behave like a spatial CNN.

4. **Augmentation is separate.** Rotate / scale / shift already fight absolute
   position. Apply geometry on the pixel grid **first**, then apply the fixed
   pack map. Packing does not replace aug.

5. **Antipodal pool stays a bad fit for image demos.** Under Cartesian +
   plane-bit packing, the antipode \(v \oplus (2^{\mathrm{DIM}}-1)\) flips
   *all* bits → scrambled \((y,x)\) **and** plane, not “2×2 spatial pool.”
   Locality packing **reinforces** keeping **no pool** (or designing a
   different, bit-local pool — §7 Phase 3).

---

## 4. Families of solutions

### A. Cartesian product packing (primary recommendation)

**Idea:** 32×32 × 2 views → 11-bit address with Gray \(x\), Gray \(y\), plane bit.

**Encode (reference):**

```text
v = (plane << 10) | (gray5(y) << 5) | gray5(x)
plane ∈ {0,1},  y,x ∈ {0,…,31}
```

**Pipeline:**

1. Build 32×32 ink (and \|∇\|) as today.  
2. For each \((y,x)\): scatter `ink[y,x]` → `out[enc(y,x,0)]`,  
   `grad[y,x]` → `out[enc(y,x,1)]`.  
3. Precompute a LUT of size 1024 (or 2048) at process start.

**Pros**

- Exact fit to \(2^5 \times 2^5 \times 2\).
- Bit groups = spatial / view axes.
- Same-location ink↔grad via one bit.
- Fast inverse; trivial tests.
- Interpretable \(w_k\) after training (optional analysis).

**Cons**

- Borders and Gray seams need explicit tests.
- Not rotation-invariant by construction (aug still required).
- Assumes power-of-two plane size (already true: 32).

**Expectation:** best **effort / insight** ratio. Highest chance that *conv*
actually uses geometry.

---

### B. Hierarchical / multi-scale bit fields (strong variant of A)

Do not treat all 5+5 bits as equal fine-grid axes only. Examples:

```text
y: 2 coarse bits + 3 fine bits
x: 2 coarse bits + 3 fine bits
plane: 1 bit
```

Or **bit-interleaved** coordinates (Morton-like ordering of axis bits while
still knowing which bits belong to \(x\) vs \(y\)).

**Why this is on-model**

- Fine bits ≈ local spatial steps.  
- Coarse bits ≈ large jumps (quadrants, half-planes).  
- Stacked layers build multi-scale receptive fields along bit composition.

**Implementation note:** HypercubeCNN tiles vertices with `T = 64` and keeps
low-bit flips inside a tile. Putting **fine spatial structure on low bits**
can align **geometry with cache behavior** — a rare case where locality packing
and performance co-design are the same decision.

**Expectation:** more design care; possible better multi-scale features than
flat Gray fields.

---

### C. Curve packing (Hilbert → 10-bit Gray + plane bit)

**Pros:** standard locality literature; good average clustering.  
**Cons:** bit directions uninterpretable; weaker match to factored kernels.  
**Use when:** Cartesian is implemented and measured; need an independent
locality prior for ablation.

---

### D. Locality in channels, not only vertices (vision-native design)

Channels in HypercubeCNN are independent copies of the **same** vertex
geometry (`activations[c * N + v]`).

**Alternative ontology (standard CNN):**

| Role | Representation |
|------|----------------|
| Space | hypercube vertices (Cartesian Gray on DIM = 10, \(N = 1024\)) |
| Feature maps / views | **channels**: ink, \|∇\|, optional blur / Laplacian, … |

Then conv1 mixes views with **shared spatial geometry**, as in every multi-
channel vision net.

| | Current: DIM=11, 1 ch, two halves of \(N\) | Alt: DIM=10, ≥2 ch, one plane |
|--|---------------------------------------------|-------------------------------|
| Spatial bits | 10 + 1 plane bit in the address | 10 pure spatial |
| View mixing | plane bit and/or FLATTEN | true channel mix in conv |
| Head (no pool) | \(C \times 2048\) | \(C \times 1024\) |
| Code surface | pack only | `input_channels`, pack shape, first layer |

The “two halves of \(N\)” pack was a clever **occupancy** trick for
single-channel demos. It is **not** the most locality-friendly long-term
story for images.

**Expectation:** more than a scatter-LUT change; cleaner science; possibly
better use of conv than stuffing views into vertex IDs.

---

### E. Overcomplete / redundant packing

Embed the same 32×32 content **multiple** ways (Cartesian, Hilbert, transpose)
into different bit regions or channels.

- **Pros:** model can select a useful embedding; robust.  
- **Cons:** burns capacity; interpretability suffers; overfit risk.  
- **Use:** research ablation, not default demo.

---

### F. Learned packing (not recommended initially)

| Approach | Issue |
|----------|--------|
| Discrete bijection pixel→vertex | Combinatorial; not cleanly differentiable w.r.t. index identity under FLATTEN |
| Soft assignment pixel→distribution over verts | Differentiable but densifies input; pipeline change |

**Verdict:** paper-scale research after A–D are measured.

---

### G. What not to do

| Idea | Why skip |
|------|----------|
| Random fixed permutation as “the” pack | Control ablation only |
| 1D snake without Gray | Weak Hamming locality |
| Expect antipodal MAX to become spatial 2×2 | Geometry does not match; already hurts MNIST |
| Expect packing alone to replace geometric aug | Different jobs |
| Non–power-of-two planes without a pad/crop policy | Breaks clean Cartesian story |
| Silent DIM changes without updating pack math | Easy to ship a nonsense map |

---

## 5. Interaction with the current training recipe

| Component | Interaction |
|-----------|-------------|
| **Train aug** (rot / scale / shift / noise) | Warp on 28×28 (or 32×32) **first**; pack is a pure function of grid coordinates afterward. |
| **\|∇\| plane** | Compute on the same 32×32 grid **before** scatter; same \((y,x)\) → same spatial bits, different plane or channel. |
| **No antipodal pool** (current default) | Keeps full \(N\); packing benefits remain visible through to FLATTEN. |
| **FLATTEN readout** | Still position-addressable. Packing helps mainly via better early features. To *see* packing effect, optional probes: fewer epochs curve, or temporarily smaller head / spatial bit-pool. |
| **Width 16, Adam, cosine, wd** | Unchanged for Phase 1 drop-in; multi-channel DIM=10 may need a light LR/wd check. |
| **Weight seeds** | Keep multi-seed discipline (`398479293` default; quote **mean**). |

**Subtle point:** with a strong FLATTEN head, packing may improve **early
optimization** (epochs 1–20) more than the final 99.2x plateau. Log full
curves, not only best-acc.

---

## 6. Pack quality metrics (before training)

Define geometry scores independent of MNIST accuracy. Over all 4-neighbors
\((p,q)\) on the 32×32 grid (and optionally both planes):

1. **Mean Hamming distance** \( \mathbb{E}[d_H(v_p, v_q)] \).  
2. **Fraction with \(d_H = 1\)** (primary score).  
3. **Histogram of \(d_H\)**.  
4. **Same-site cross-view distance:** for ink vs grad at equal \((y,x)\),  
   \(d_H(v_{\mathrm{ink}}, v_{\mathrm{grad}})\) — should be **1** (plane bit only)
   under Cartesian plane packing.

| Pack | Expected 4-neigh \(d_H=1\) rate | Same-site ink↔grad |
|------|----------------------------------|--------------------|
| Row-major halves | low | often large (not 1) |
| Cartesian Gray + plane | **high** (design target) | **1** |
| Hilbert + Gray + plane | medium–high average | **1** if plane bit separate |

If Cartesian Gray does not dominate row-major on (2) and (4), the encoder is
wrong — fix before any training A/B.

---

## 7. Recommended implementation path

### Phase 0 — Metrics harness

- Implement Gray5 encode/decode.  
- Implement row-major vs Cartesian scatter.  
- Print pack metrics (§6).  
- Unit test: round-trip \((y,x,\mathrm{plane}) \leftrightarrow v\);  
  unit step in \(x\) or \(y\) flips exactly one bit in the right field.

### Phase 1 — Drop-in pack (minimal risk)

Keep **DIM=11, 1 input channel, length-2048 vector**, same net and schedule.

- Replace only the scatter inside `pack_mnist_2048` (or equivalent).  
- Flag or compile-time choice: `RowMajor` vs `CartesianGray`.  
- Precomputed LUT `[2][32][32]` or flat 2048.

**A/B protocol:**

- Same aug, 60 epochs, batch 256, lr/wd as documented.  
- At least seed `398479293`; prefer full 3-seed mean when claiming wins.  
- Report pack metrics + best-acc / best-loss / curve.

**Success criteria (reasonable):**

- Throughput ≈ unchanged (LUT scatter is negligible vs train).  
- Mean best-acc **≥** baseline within noise, or **≥ +0.05 pp**.  
- Clear win on pack metrics.  
- Optional: faster climb to 99%.

### Phase 2 — Multi-channel spatial cube (vision-native)

- DIM = 10, \(N = 1024\), `input_channels = 2` (ink, \|∇\|).  
- Cartesian Gray on 10 bits only.  
- First conv sees `c_in = 2`.  
- Document new param counts and head size (\(C \times 1024\) if no pool).

Treat as a **separate** experiment (not a silent default swap).

### Phase 3 — Hierarchical bits + spatial-style pool (optional)

If Phase 1 helps:

- Coarse/fine bit fields (§4B).  
- **Pool = reduce along one fine spatial bit** (max/avg of \(v\) and
  \(v \oplus 2^{k_{\mathrm{fine}}}\)), i.e. honest 2:1 downsample on one
  Gray axis — **not** antipodal pool.

That is the hypercube analogue of spatial pooling that respects packing.

### Phase 4 — Ablations only

- Hilbert pack.  
- Binary (non-Gray) Cartesian.  
- Overcomplete dual embed.  
- Avoid learned packing until the above are boring.

---

## 8. Out-of-the-box ideas that remain grounded

1. **Plane bit as multi-view edge** under Cartesian packing — first-class,
   not a hack.  
2. **Channels = views, vertices = space** — standard vision ontology on a
   cube (Phase 2).  
3. **Bit order co-designed with `T=64` tiles** — geometry + L1 behavior.  
4. **Diagnostic, not just accuracy:** compare \(\|w_k\|\) (or average
   |kernel| per direction) for Cartesian vs row-major. Cartesian should show
   more structure across \(x\)-bits vs \(y\)-bits; row-major should look
   flatter/messier.  
5. **Fixed pack + learned 1×1 channel mix** after multi-channel pack — cheaper
   than learned vertex assignment.  
6. **Locality pack + deliberately weaker head** (fine-bit pool once) — science
   experiment: can geometry carry more load? May cost peak acc temporarily.

---

## 9. What not to claim

- That locality packing is **required** to justify HypercubeCNN (native
  hypercube data never needed an image map).  
- That it will **definitely** beat elastic/shear aug for the next 0.1 pp.  
- That Hilbert is “the” solution because it is popular for caches.  
- That antipodal pooling becomes appropriate once packing is local.  
- That row-major was “wrong” for the ~99.3% result — it was a valid
  occupancy + multi-view engineering choice.

---

## 10. Bottom line

**Best conceptual fit for this architecture:**

> Treat the hypercube as a **binary product space for a power-of-two image**,
> with **Gray-coded spatial axes** and **extra bits for views or scale** —
> not as a random address space filled in row-major order.

**Best first implementation:**

| Priority | Design |
|----------|--------|
| **1** | **Cartesian Gray 5+5 + plane bit** on current DIM=11, 1-channel, 2048 pack (drop-in) |
| **2** | **DIM=10, 2+ channels** (ink, grad), pure spatial cube |
| **3** | Hierarchical bits; fine-bit spatial pool; Hilbert ablation |
| **Skip for now** | Learned packing; antipodal-as-spatial; overcomplete multi-embed as default |

**Reasonable expectation:** clearer use of Hamming convolution and a fairer
scientific story for “images on a hypercube”; **modest** accuracy movement on
an already ~99.3% (single-seed) recipe. The primary win is **alignment and
interpretability**; leaderboard gains are secondary and must be measured.

**Implementation surface for Phase 1:** example/demo pack code only (e.g.
`examples/mnist_train.cpp` or a small shared pack helper) — **no core SDK
change** required for the drop-in Cartesian scatter.

---

## 11. Reference sketch (Phase 1 encoder)

Illustrative only; not production code.

```cpp
// 5-bit binary-reflected Gray
static inline unsigned gray5(unsigned u) {
    u &= 31u;
    return u ^ (u >> 1);
}

// plane: 0 = ink, 1 = |grad|
static inline unsigned cartesian_vertex(unsigned y, unsigned x, unsigned plane) {
    return (plane << 10) | (gray5(y) << 5) | gray5(x);
}

// Scatter 32x32 ink and grad into length-2048 channel-0 buffer
static void pack_cartesian_gray(const float* ink32, const float* grad32, float* out2048) {
    for (unsigned y = 0; y < 32; ++y) {
        for (unsigned x = 0; x < 32; ++x) {
            const unsigned i = y * 32 + x;
            out2048[cartesian_vertex(y, x, 0)] = ink32[i];
            out2048[cartesian_vertex(y, x, 1)] = grad32[i];
        }
    }
}
```

Pair with the metrics in §6 and the A/B protocol in §7 Phase 1 before
changing documented default results in `mnist_train.md`.

---

## Related docs

| Doc | Role |
|-----|------|
| `examples/mnist_train.md` | Current demo recipe, aug, multi-seed results |
| `examples/mnist_train.cpp` | Pack + aug + train loop implementation |
| `docs/architecture.md` | Hypercube conv, antipodal pool, FLATTEN, tiling |

---

*This memo is image-demo engineering guidance. Native hypercube applications
(fingerprints, Boolean functions, reservoir state) do not require a 2D
locality map; their vertex layout is already the domain geometry.*
