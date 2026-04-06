# MoleculeNet Baselines & HCNN Results

Scaffold split, AUC-ROC. Compiled 2026-04-06.

Sources: MoleculeNet (Wu et al. 2018), OGB (Hu et al. 2020), Uni-Mol (Zhou et al. 2023), UniMol+ (Lu et al. 2024).
OGB leaderboard (`ogbg-molhiv`) is the authoritative HIV reference.

---

## BBBP (Blood-Brain Barrier Permeability)

1,631 train / scaffold split / binary classification

| Method | Type | AUC-ROC |
|--------|------|---------|
| MLP (ECFP) | Fingerprint | ~0.67 |
| GCN | GNN | ~0.69 |
| RF (ECFP) | Fingerprint | ~0.71 |
| MPNN | GNN | ~0.71 |
| **HCNN** | **Hypercube CNN** | **0.833** |
| HCNN (no-conv ablation) | Density baseline | 0.815 |

HCNN beats all baselines. Ablation shows density is a strong predictor on this dataset (0.815 without conv), but conv adds +0.02.

---

## BACE (BACE-1 Inhibition)

1,210 train / scaffold split / binary classification

| Method | Type | AUC-ROC |
|--------|------|---------|
| MLP (ECFP) | Fingerprint | ~0.67 |
| RF (ECFP) | Fingerprint | ~0.68 |
| **HCNN** | **Hypercube CNN** | **0.747** |
| GCN | GNN | ~0.78 |
| MPNN | GNN | ~0.81 |

HCNN beats fingerprint baselines but lags behind graph methods. No ablation run yet.

---

## HIV (HIV Activity)

32,896 train / scaffold split / binary classification / 3% positive (extreme imbalance)

| Tier | Method | Type | AUC-ROC |
|------|--------|------|---------|
| SOTA | UniMol+ | 3D pre-trained | ~0.82-0.84 |
| SOTA | Uni-Mol, GROVER, GEM | Pre-trained / geometry | ~0.80-0.82 |
| Strong | ChemBERTa-2 | SMILES transformer | ~0.78-0.80 |
| Strong | RF (ECFP) | Fingerprint | ~0.78-0.80 |
| Strong | AttentiveFP | Graph attention | ~0.78-0.79 |
| Strong | D-MPNN (Chemprop) | Directed MPNN | ~0.77-0.78 |
| Baseline | GIN + virtual node | GNN | ~0.77-0.78 |
| Baseline | GCN | GNN | ~0.76-0.78 |
| Baseline | MPNN | GNN | ~0.77 |
| Baseline | MLP (ECFP) | Fingerprint | ~0.76-0.78 |
| | **HCNN (current)** | **Hypercube CNN** | **~0.72** |

### Notes

- RF on ECFP is extremely competitive on HIV; many GNN papers only marginally exceed it.
- Models that clearly beat fingerprints use self-supervised pre-training on millions of molecules or 3D conformer information.
- Variance is high due to 3% positive rate. 1-2% AUC differences can be within noise. OGB requires multi-seed reporting.
- **Realistic HCNN target: 0.78-0.80** (match fingerprint-based classical ML). Anything above requires pre-training or 3D data.

---

## HCNN Architecture Summary

| Dataset | Size | Architecture | Params | Config |
|---------|------|-------------|--------|--------|
| BBBP | <5K | 2 conv+pool (16->32) | ~5K | lr=0.03, wd=1e-3, 60 ep |
| BACE | <5K | 2 conv+pool (16->32) | ~5K | lr=0.03, wd=1e-3, 60 ep |
| HIV | >5K | 4 conv+pool (32->64->128->128) | ~200K | lr=0.03, wd=1e-4, 40 ep |

All: bipolar input (0->-1, 1->+1), cosine LR, momentum 0.9, batch 32, DIM=10 (1024 vertices).
HIV uses minority oversampling (positives resampled to match negative count).
