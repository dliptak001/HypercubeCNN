# Prior Art Survey: CNN on Hypercube Substrate

Survey conducted 2026-04-04. Searched Google Scholar, arXiv, and general web for prior work on convolutional neural networks operating on binary hypercube vertices with Hamming-distance-indexed kernels.

## Conclusion

No prior work combines all elements of this architecture: data embedded onto 2^DIM hypercube vertices, convolution with K=DIM nearest-neighbor XOR masks (one learned weight per Hamming-distance-1 neighbor), antipodal pooling by dimension reduction, and a GAP-or-FLATTEN linear readout to class logits. The mathematical ingredients exist separately in the literature, but nobody has assembled them into a CNN architecture trained end-to-end with backpropagation.

---

## Closely Related Work

### EHCube4P (Daud, Charton, Damour, Wang, Cadet, 2025)

- **Venue:** arXiv (q-bio.QM)
- **Link:** https://arxiv.org/abs/2506.16921
- **Summary:** Models protein mutation sequences as vertices of a hypercube and runs a GCN on this graph. Uses standard Kipf-Welling GCN (normalized adjacency), not Hamming-distance-indexed kernel weights. Edge weights are based on bitwise dot product, creating implicit distance-dependent weighting, but this is not the same as explicit shared kernel [w0..wr]. No pooling by dimension reduction.
- **Distinction:** Same substrate (hypercube graph), different convolution mechanism. Treats the hypercube as a generic graph for standard GCN, not as a structured domain with distance-based weight sharing.

---

## Moderately Related Work

### KP-GNN (Feng et al., NeurIPS 2022)

- **Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2205.13328
- **Summary:** K-hop message passing where each hop distance gets its own shared parameter set. Architecturally the closest general GNN concept to our distance-indexed kernel. However, designed for arbitrary graphs, does not exploit hypercube structure (vertex-transitivity, bitwise geometry, XOR enumeration). No hypercube embedding or dimension-reducing pooling.
- **Distinction:** Same "shared weights per hop distance" principle, but on generic graphs without hypercube structure.

### Group Equivariant Convolutional Networks (Cohen & Welling, ICML 2016)

- **Venue:** ICML 2016
- **Link:** https://arxiv.org/abs/1602.07576
- **Summary:** Defines convolution over discrete symmetry groups (G-convolutions). The Boolean hypercube Z_2^n is an abelian group, and convolution over it is the Walsh-Hadamard transform. This framework could in principle be instantiated on Z_2^n, but the paper never does so -- it focuses on p4, p4m groups on the 2D grid. Our architecture additionally constrains the kernel to be isotropic (same weight for all neighbors at the same Hamming distance), which is stricter than general group convolution on Z_2^n.
- **Distinction:** Provides the mathematical framework that could encompass our approach as a special case, but never considers the hypercube domain or distance-based isotropy.

### Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges (Bronstein et al., 2021)

- **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2104.13478
- **Summary:** Comprehensive unifying framework treating standard CNNs, GNNs, and group-equivariant networks as instances of a single blueprint. The Boolean hypercube as a Cayley graph of Z_2^n fits within their group convolution framework, but they do not discuss it explicitly. No mention of Hamming-distance kernels or hypercube substrates.
- **Distinction:** The theoretical umbrella that covers our architecture exists, but nobody has instantiated it on the hypercube with distance-isotropic kernels.

### Spherical CNNs (Cohen, Geiger, Koehler, Welling, ICLR 2018)

- **Venue:** ICLR 2018
- **Link:** https://arxiv.org/abs/1801.10130
- **Summary:** Convolution on the sphere S2 and rotation group SO(3) using spherical harmonics. Closest analog in spirit: isotropic convolution on a non-Euclidean domain, where kernels depend only on geodesic distance, just as our hypercube kernels depend only on Hamming distance. However, the domain is continuous (sphere) rather than discrete (hypercube).
- **Distinction:** Same design philosophy (isotropic convolution on non-Euclidean domain), different domain (continuous manifold vs discrete combinatorial structure).

### Learning Boolean Functions with Neural Networks (Abbe, Boix-Adsera, Misiakiewicz, ~2023)

- **Venue:** OpenReview (ICLR/NeurIPS submission)
- **Link:** https://openreview.net/forum?id=LEuuOaZNOT
- **Summary:** Studies gradient descent on 2-layer fully-connected networks learning functions f: {0,1}^n -> R on the Boolean hypercube. Shows spectral bias: lower-degree Fourier components are learned first. Same domain (Boolean hypercube as input space), but no convolution, no weight sharing, no graph structure exploitation.
- **Distinction:** Same domain, but standard MLP architecture. Purely a learnability theory result.

### CayleyNets (Levie et al., 2017)

- **Venue:** arXiv
- **Link:** https://arxiv.org/abs/1705.07664
- **Summary:** Spectral graph convolution using Cayley polynomials. Since the hypercube is the Cayley graph of Z_2^n, CayleyNets could theoretically be applied to it. Spectral filters on the hypercube would correspond to Walsh-Hadamard domain operations, related to but distinct from spatial Hamming-distance kernels.
- **Distinction:** Relevant spectral machinery, not applied to hypercubes.

---

## Tangentially Related Work

### HyperNEAT (Stanley, D'Ambrosio, Gauci, 2009)

- **Venue:** Artificial Life (MIT Press)
- **Link:** https://direct.mit.edu/artl/article-abstract/15/2/185/2634/
- **Summary:** Uses a hypercube as a coordinate space for placing neurons. "Hypercube" refers to the encoding geometry, not a computation graph. No Hamming distance, no convolution on vertices.
- **Distinction:** Unrelated despite the name.

### Walsh-Hadamard Transform in Neural Networks (various, 2021+)

- **Link:** https://arxiv.org/abs/2104.07085
- **Summary:** Replaces 1x1 convolution layers with WHT-based binary layers for efficiency. The WHT is the Fourier transform on Z_2^n, so there is a deep mathematical connection: a Hamming-distance kernel in the spatial domain corresponds to a diagonal operator in the WHT domain. However, these papers use WHT purely as a fast transform, not as convolution on a hypercube substrate.
- **Distinction:** Same mathematical transform, completely different purpose.

### Talagrand's Convolution Conjecture (Chen, 2025)

- **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2511.19374
- **Summary:** Proves that convolution on the Boolean hypercube has a natural regularization/smoothing effect (30-year-old conjecture). Not a neural network architecture, but provides theoretical justification for why hypercube convolution might generalize well.
- **Distinction:** Theoretical support, not an architecture.

### Hyperdimensional Computing (Kanerva, various years)

- **Link:** https://en.wikipedia.org/wiki/Hyperdimensional_computing
- **Summary:** Brain-inspired computing using high-dimensional binary vectors. Vertices in binary vector space can be viewed as hypercube vertices with Hamming distance, but the operations (XOR binding, majority bundling) are fundamentally different from convolution. No learned kernels, no distance-based weight sharing.
- **Distinction:** Same mathematical object (binary hypercube), completely different computational paradigm.

---

## What Is Novel in HypercubeCNN

The specific combination of the following elements appears in no prior work:

1. **Embedding data onto Boolean hypercube vertices** (input mapped to 2^DIM nodes via Direct Linear Assignment)
2. **Convolution with K=DIM nearest-neighbor XOR masks** -- each mask is a single-bit flip (Hamming distance 1), one learned weight per neighbor direction, shared across all vertices
3. **Bitwise geometry via XOR** -- neighbor lookup is a single XOR per mask, no adjacency lists, no spatial indexing, no padding, no border handling
4. **Multi-channel convolution** with independent kernels per output channel, optional per-channel batch normalization, and configurable activation (NONE / ReLU / LeakyReLU)
5. **Antipodal pooling by dimension reduction** (pairing each vertex with its bitwise complement, preserving exact hypercube geometry; MAX or AVG)
6. **Two readout strategies**: global average pooling per channel (translation-invariant) or full-flatten (position-sensitive), each followed by a linear layer to class logits
7. **End-to-end backpropagation** with SGD-momentum or Adam (decoupled weight decay), per-network configurable
