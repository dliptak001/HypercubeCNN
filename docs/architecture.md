# HypercubeCNN Architecture

## Conceptual Overview

1. **Vertices** — the only spatial primitive
   - Fixed number of vertices `N = 1 << DIM`.
   - Every vertex holds a scalar `float` activation.
   - All geometry is defined by Hamming distance (bit flips). No adjacency lists, no padding, no borders.

2. **Channels**
   - Multiple independent hypercubes sharing the exact same vertex set.
   - Stored as one flat array: `C x N` floats.

3. **Conv layer (`HCNN`)**
   - Shared kernel: a tiny array of weights `[w0, w1, ..., wr]` (one weight per Hamming distance).
   - Each output channel at each vertex computes a weighted sum over input-channel shells.
   - Activation (ReLU) + optional bias applied per output channel.

4. **Pooling layer (`HCNNPool`)**
   - Reduces `DIM` by `reduce_by` via max or average over subcubes.
   - Output is a perfect smaller hypercube — geometry preserved.

5. **Stacking (`HCNNNetwork`)**
   - Sequence of conv + pool layers.
   - Dimension shrinks only on pool layers.
   - Final feature volume: `C_final x N_final` floats.

6. **Readout (`HCNNReadout`)**
   - Global average per final channel -> linear layer to class logits.
   - Fixed-size output (e.g. 10 for MNIST).

Everything operates on flat 1D arrays. No image concepts inside the core.

## Input Embedding

This is the only place where Euclidean-structured scalar data (pixels, patches, flattened tensors) meets the hypercube.

### Goal

Map an arbitrary-length list of input scalars (e.g. 784 MNIST pixels) onto the `N` hypercube vertices so that:

- Spatial structure from the original data is preserved as much as possible.
- The hypercube's vertex-transitive symmetry and tiny diameter (only `DIM` hops to anywhere) can still be leveraged.
- No new data structures, no complexity, no dependencies.

### Core insight

Because the hypercube diameter is extremely small, *any reasonable assignment* of scalars to vertices will allow information to propagate globally in just a few conv layers. The embedding does not need to be perfect — it only needs to be simple and repeatable.

### Options (ranked by simplicity)

#### Option A — Direct Linear Assignment (default)

- Flatten the input data into a 1D array of scalars.
- Assign the first `min(input_length, N)` scalars directly to vertices `0 ... min-1` (in vertex-index order).
- If input is shorter than `N`, pad the remaining vertices with zero (or a learned constant).
- If input is longer than `N`, either truncate or fold (sum/modulo) the excess into the vertices.
- **Why it works:** The hypercube's fast connectivity means local Euclidean neighborhoods quickly become Hamming-neighborhood mixtures after one or two layers. No hashing, no binary encoding — pure scalars on vertices.

#### Option B — Recursive Subcube Assignment (preserves hierarchy)

- Treat the hypercube as recursively composed of subcubes.
- Assign input patches to entire subcubes (e.g. first `2^k` vertices get the first patch).
- This naturally mirrors the hierarchical pyramid we already get from pooling.
- Slightly more structure than Option A, still zero runtime cost.

#### Option C — Bit-Interleaved / Gray-Code Ordering (locality bias)

- Order the vertices using a Gray-code or bit-interleaving scheme so that spatially close input pixels tend to land on vertices with small Hamming distance.
- More sophisticated, but still O(N) and constant-time.
- Only worth it if experiments show Option A loses too much locality.

### Architectural rule

The embedding lives **outside** `HCNNNetwork`. A thin helper function takes the raw scalar input array and produces the first-layer activation array (`1 x N` or `C_in x N`). The network itself remains completely agnostic — it only ever sees a flat activation array on the hypercube.
