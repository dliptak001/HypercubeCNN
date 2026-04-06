# Boolean Functions on the Hypercube

## The setup

A Boolean function is simply f: {0,1}^n → {0,1}. The input is an n-bit binary vector — a vertex of the n-dimensional hypercube. The output is a single bit.

This is the purest possible test for HCNN. There's no embedding, no distortion, no spatial structure being destroyed. The input IS a hypercube vertex. If the architecture can't win here, it can't win anywhere.

## Why parity is the killer test

The **parity function** outputs 1 if the number of set bits is odd, 0 if even. Equivalently, it's the XOR of all input bits:

```
parity(x) = x_0 ⊕ x_1 ⊕ x_2 ⊕ ... ⊕ x_{n-1}
```

This function is notoriously hard for MLPs. Here's why:

**MLPs compute parity by accident, not by design.** An MLP with ReLU activations computes piecewise-linear functions. To represent parity on n bits, it needs to carve {0,1}^n into 2^n regions (every vertex gets a different sign from at least one neighbor). A single hidden layer needs width proportional to 2^(n-1). Even deep MLPs need total parameters proportional to 2^n. This is a proven lower bound — not a matter of training tricks.

**HCNN computes parity by structure.** Look at what parity means geometrically on the hypercube: every vertex has the opposite parity from all its Hamming-distance-1 neighbors (flipping any single bit changes parity). This is exactly what our K=DIM nearest-neighbor kernel sees. A single HCNN conv layer with 1 output channel could, in principle, learn: "if the weighted sum of my neighbors disagrees with me, I'm at a parity boundary." The architecture's inductive bias is aligned with the function's structure.

## The family of test functions

Parity is one extreme. Here's the full spectrum, from HCNN-friendly to HCNN-neutral:

**Parity (k-bit)**: XOR of k specific bits. When k=n, this is full parity. When k=3, it's a 3-bit XOR embedded in n dimensions. The key property: the function value changes whenever any of the k participating bits flips. This creates a checkerboard pattern on the hypercube that perfectly matches Hamming-distance-1 convolution.

**Threshold functions**: output 1 if the Hamming weight (number of set bits) is >= k. For example, "majority" outputs 1 if more than half the bits are set. These functions are *constant* on Hamming shells — all vertices with the same Hamming weight have the same output. This is exactly the symmetry our convolution respects (distance-indexed kernels treat all same-distance neighbors identically).

**DNF formulas**: a k-term Disjunctive Normal Form, like `(x_1 ∧ x_3 ∧ x_7) ∨ (x_2 ∧ x_5)`. Each term is a conjunction (AND) of a few variables. The output is 1 if any term is satisfied. These have localized structure on the hypercube — each term defines a subcube corner, and the OR creates a union of subcube neighborhoods. HCNN should learn these through multi-layer feature composition.

**Random Boolean functions**: a random truth table over 2^n entries. No structure at all. No method should do well without memorizing the whole table. This is the control — HCNN and MLP should perform equally (both just memorize).

## What the experiment looks like

For each function class at DIM=10 (1024 vertices):

1. **Generate the truth table**: all 1024 input/output pairs are known exactly.
2. **Split**: train on a random subset (say 50%, 70%, 90% of vertices), test on the rest.
3. **Train HCNN and MLP** with the same parameter budget.
4. **Measure**: accuracy on held-out vertices (generalization), epochs to converge, training set size needed for 99% accuracy (sample efficiency).

The generalization question is the interesting one: given 50% of the truth table, can the model predict the other 50%? An MLP has no reason to generalize — every vertex is independent to it. But HCNN knows that nearby vertices (Hamming distance 1) should be processed similarly. For parity, that's exactly the right bias: if you know the parity of vertex v, you know the parity of all its neighbors (it's flipped).

## The theoretical argument

On Z_2^n (the Boolean hypercube as a group), any Boolean function has a unique **Fourier expansion** over the Walsh-Hadamard basis. The Fourier coefficients at each "frequency" correspond to subsets of input bits that interact.

Parity has its entire energy at the highest frequency (all bits participate). Low-degree functions (like threshold-1 or single-term ANDs) have energy at low frequencies.

Our Hamming-distance-1 convolution is essentially a graph filter on the hypercube's Cayley graph. It naturally captures interactions between adjacent vertices — which corresponds to specific frequency bands in the Walsh-Hadamard spectrum. This is the theoretical reason HCNN should excel: its convolution is a natural spectral filter on the hypercube, whereas an MLP has to learn the spectral structure from scratch.

## What success looks like

| Function | MLP (expected) | HCNN (expected) | Why |
|----------|---------------|-----------------|-----|
| Parity (full) | Fails without exponential width | Learns with O(n) parameters | Kernel aligned with parity structure |
| Parity (k=3) | Needs ~2^3 hidden units | Single layer sufficient | 3-bit interaction = distance-3 pattern |
| Majority | Moderate — linear separator exists | Easy — threshold on Hamming weight | Convolution sees Hamming shells directly |
| DNF (k terms) | Moderate | Good — multi-layer feature detection | Subcube corners are local in Hamming distance |
| Random | Both memorize | Both memorize | No structure to exploit |

The parity result would be the headline: "HCNN learns full parity on n=10 bits from 50% of the truth table; MLP with equal parameters cannot." That's a clean, reproducible, theoretically grounded result that demonstrates the architecture's core advantage.

## Why this matters beyond Boolean functions

If HCNN can learn structured Boolean functions from partial truth tables, it proves the architecture has the right inductive bias for **any data where Hamming-distance relationships carry meaning**. That's the bridge to molecular fingerprints — fingerprint similarity IS Hamming distance, and property prediction IS learning a (noisy) function over those fingerprints.

Boolean functions are the clean, synthetic version of the molecular fingerprint problem. If the bias works here, we have strong reason to believe it works there.
