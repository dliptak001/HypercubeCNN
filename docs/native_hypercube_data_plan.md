# Native Hypercube Data — Research Plan

Goal: demonstrate HypercubeCNN on data that naturally lives on binary hypercubes,
where the architecture's inductive bias (Hamming-distance isotropy) is an advantage
rather than a handicap.

---

## Track A: Boolean Functions — Abandoned

**Status**: abandoned after investigation revealed fundamental task mismatch.

Boolean function learning (predict f(v) for a single vertex v) is a **point classification** task. HypercubeCNN is a **field classification** architecture — it processes activation maps (values at every vertex), not individual coordinates. This mismatch required increasingly elaborate input encodings, each a workaround rather than native use of the architecture.

See [boolean_functions.md](boolean_functions.md) for full analysis.

**Key takeaway**: "input is a hypercube vertex" does not mean "input is native hypercube data." Native hypercube data is a **vector of values across all vertices** — a field, not a point.

---

## Track B: Molecular Fingerprints (paper-worthy)

**Timeline**: 1 week implementation, 1-2 weeks experiments + writing

**Pitch**: Molecular fingerprints (ECFP, MACCS keys) are binary vectors — literally
activation maps over hypercube vertices. Each bit indicates the presence of a molecular
substructure. The full vector IS the input, and the task is to classify the **pattern**.
This is exactly what a CNN is designed to do.

Existing methods treat fingerprints as flat feature vectors (MLP, RF) or
convert back to molecular graphs (GNN). HCNN is the first model that respects their
native Hamming geometry.

### B1. Data pipeline
- **Datasets**: Tox21, BBBP, HIV, BACE from MoleculeNet benchmark suite.
  All have standard train/val/test splits and published baselines.
- **Featurization**: use RDKit to compute ECFP fingerprints (1024 or 2048 bits).
  Each molecule becomes a single binary vector over the hypercube (DIM = 10 or 11).
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

1. **Done**: MNIST baseline (~98%) validates architecture + training pipeline.
2. **Done**: Boolean function investigation — clarified field vs point classification distinction.
3. **Next**: Molecular fingerprints. The real native test case.
4. **Write-up**: molecular fingerprint results + MNIST as supplementary + Boolean functions as methodology note (what we learned about matching tasks to architectures).

---

## Success criteria

- HCNN beats RF and MLP on at least 3 of 4 MoleculeNet tasks (AUC-ROC).
- Conference-quality result if achieved with standard fingerprints and no task-specific tuning.
