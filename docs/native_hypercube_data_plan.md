# Native Hypercube Data — Research Plan

Goal: demonstrate HypercubeCNN on data that naturally lives on binary hypercubes,
where the architecture's inductive bias (Hamming-distance isotropy) is an advantage
rather than a handicap.

Two tracks, in priority order.

---

## Track A: Boolean Function Learning (proof-of-concept)

**Timeline**: 1-2 days implementation, 1 day experiments

**Pitch**: HCNN's convolution kernel is indexed by Hamming distance — exactly the
right inductive bias for functions on Z_2^n. Show it learns Boolean functions faster
and with fewer parameters than an MLP of equal capacity.

### A1. Synthetic data generator
- Generate training samples for target Boolean functions of DIM variables.
- Each sample: input = DIM-bit vector (a hypercube vertex), output = f(input).
- Functions to test (increasing complexity):
  - **Parity**: XOR of k bits (k = 3, 5, 7, DIM). Known to be hard for MLPs.
  - **Majority**: output 1 if more than DIM/2 bits are set.
  - **Threshold-k**: output 1 if Hamming weight >= k.
  - **Random monotone**: random monotone Boolean function (generated once, fixed).
  - **DNF formulas**: k-term DNF with t literals per term.
- No embedding needed — input IS a vertex. This is the ideal case.

### A2. HCNN configuration for Boolean functions
- Binary classification (num_classes = 2).
- DIM = 8, 10, 12, 14 to show scaling.
- Vary depth (1-4 conv layers) and width (16-128 channels).
- No pooling (or test with/without to measure its contribution).

### A3. Baselines
- **MLP**: same parameter count, ReLU, same optimizer. This is the Abbe et al. comparison.
- **GNN (GCN/GAT)**: hypercube graph as adjacency, same parameter budget.
  Requires PyTorch Geometric or similar — could be a Python-side comparison.
- **Random forest / XGBoost**: strong non-neural baseline for tabular binary data.

### A4. Metrics
- **Sample efficiency**: accuracy vs training set size (learning curves).
- **Generalization**: train on subset of 2^DIM vertices, test on held-out vertices.
- **Convergence speed**: epochs to 95%/99% accuracy.
- **Parameter efficiency**: accuracy vs model size.

### A5. Expected result
HCNN should crush MLPs on parity (MLPs provably need exponential width for parity;
HCNN's Hamming-distance kernel is structurally aligned). Majority and threshold
functions should also favor HCNN. If this doesn't happen, the architecture has a
fundamental problem.

---

## Track B: Molecular Fingerprints (paper-worthy)

**Timeline**: 1 week implementation, 1-2 weeks experiments + writing

**Pitch**: Molecular fingerprints (ECFP, MACCS keys) are binary vectors — literally
hypercube vertices. Existing methods treat them as flat feature vectors (MLP, RF) or
convert back to molecular graphs (GNN). HCNN is the first model that respects their
native Hamming geometry.

### B1. Data pipeline
- **Datasets**: Tox21, BBBP, HIV, BACE from MoleculeNet benchmark suite.
  All have standard train/val/test splits and published baselines.
- **Featurization**: use RDKit to compute ECFP fingerprints (1024 or 2048 bits).
  Each molecule becomes a single hypercube vertex (DIM = 10 or 11).
- **Format**: binary classification per task. Multi-task for Tox21 (12 endpoints).

### B2. HCNN configuration for fingerprints
- DIM = 10 (1024-bit fingerprints) or DIM = 11 (2048-bit).
- Architecture similar to MNIST setup but tuned for binary classification.
- Multi-task head for Tox21 (12 outputs instead of 10).

### B3. Baselines (from published MoleculeNet results)
- **Random Forest on fingerprints**: the industry standard.
- **MLP on fingerprints**: same input, no hypercube structure.
- **GCN/MPNN on molecular graph**: the current SOTA approach (different input).
- **Transformer on SMILES**: another common approach (different input).

### B4. The argument
- RF/MLP treat fingerprint bits as independent features — no structure.
- GNN/Transformer use molecular graphs/SMILES — different representation entirely.
- HCNN uses the fingerprint directly AND respects bit-flip relationships.
- If HCNN beats RF/MLP on fingerprints, that proves Hamming geometry matters.
- If HCNN is competitive with GNN, that's remarkable (fingerprints lose info vs graphs).

### B5. Dependencies
- RDKit (Python) for featurization — generates binary vectors offline.
- MoleculeNet datasets — freely available, well-documented.
- New dataloader in C++ to read pre-featurized binary vectors.
- AUC-ROC metric (standard for these benchmarks, not just accuracy).

---

## Sequencing

1. **Now**: finish MNIST accuracy push (validates architecture + training pipeline).
2. **Track A**: Boolean functions (1-2 days). Quick win, validates core thesis.
3. **Track B**: Molecular fingerprints (1-2 weeks). Paper-worthy result.
4. **Write-up**: combine Track A (theoretical validation) + Track B (applied result)
   into a single paper with MNIST as supplementary material.

---

## Success criteria

- Track A: HCNN beats MLP by >10% sample efficiency on parity functions at DIM >= 10.
- Track B: HCNN beats RF and MLP on at least 3 of 4 MoleculeNet tasks (AUC-ROC).
- Either result alone is workshop-publishable. Both together are conference-quality.
