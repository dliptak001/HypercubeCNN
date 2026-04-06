# Molecular Fingerprints as Native Hypercube Data

## What is a molecular fingerprint?

A molecular fingerprint is a fixed-length binary vector that encodes the structural features present in a molecule. Each bit position represents a specific substructure or chemical pattern — if the molecule contains that pattern, the bit is 1; otherwise 0.

For example, a 1024-bit ECFP fingerprint for aspirin might look like:
```
[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, ...] (1024 bits)
```

Bit 1 might mean "contains a carboxyl group," bit 4 might mean "contains a 6-membered aromatic ring," etc.

## How ECFP (Extended Connectivity Fingerprint) works

ECFP is the most widely used fingerprint in drug discovery. The algorithm:

1. **Start at each atom** — assign it an initial identifier based on its element, charge, bond count, etc.
2. **Iterate neighborhoods** — for each atom, collect the identifiers of all atoms within radius 1 (direct bonds), then radius 2 (two bonds away), etc. Hash each neighborhood pattern to produce an integer.
3. **Fold to fixed width** — hash each integer to a bit position in the output vector (typically 1024 or 2048 bits). Set that bit to 1.

The result: each bit encodes the *presence* of a specific local chemical environment somewhere in the molecule. Two molecules with similar substructures will have similar fingerprints — specifically, they'll be close in **Hamming distance**.

## Why fingerprints ARE hypercube vertices

A 1024-bit ECFP fingerprint is a point in {0,1}^1024 — a vertex of the 1024-dimensional binary hypercube. This isn't a metaphor; it's literally what the data is.

When chemists compare molecules, they use the **Tanimoto similarity**, which for binary vectors is closely related to Hamming distance. Two molecules with Hamming distance 5 share almost all structural features. Two molecules with Hamming distance 500 are structurally unrelated.

This means the chemical notion of "molecular similarity" is already **Hamming-distance similarity on the hypercube**. Our convolution kernel — which is indexed by Hamming distance — is operating in exactly the geometry that chemists use to reason about molecular similarity.

## How this data is currently used

In drug discovery, the standard pipeline is:

1. Compute fingerprints for all molecules (RDKit, one line of Python)
2. Feed the bit vectors into a **random forest** or **MLP**
3. Predict properties: toxicity, solubility, blood-brain barrier penetration, binding affinity

The critical gap: both RF and MLP treat the 1024 bits as **independent features**. They don't know that flipping bit 7 and flipping bit 438 might represent related structural changes. They don't know that two fingerprints at Hamming distance 3 represent chemically similar molecules.

Meanwhile, **graph neural networks** (the current SOTA) go back to the molecular graph and do message passing over atoms and bonds. They capture structure — but they throw away the fingerprint entirely and start from scratch.

Nobody convolves over the fingerprint's native Hamming geometry. That's the gap.

## What HCNN would do differently

Our convolution at DIM=10 (1024 bits): for each vertex (molecule), look at the K=10 nearest neighbors (single bit flips = molecules that differ by one structural feature) and learn weighted combinations. Stack layers to capture multi-feature interactions.

The inductive bias is: **molecules that differ by few structural features should be processed similarly**. This is exactly what medicinal chemists believe — it's why Tanimoto similarity works in the first place.

## The datasets

MoleculeNet is the standard benchmark suite. The relevant tasks:

| Dataset | Task | Molecules | Metric |
|---------|------|-----------|--------|
| **Tox21** | 12 toxicity endpoints | 8,014 | AUC-ROC |
| **BBBP** | Blood-brain barrier penetration | 2,039 | AUC-ROC |
| **HIV** | HIV replication inhibition | 41,127 | AUC-ROC |
| **BACE** | Beta-secretase inhibitor (Alzheimer's) | 1,513 | AUC-ROC |

All freely available. Published baselines for RF, MLP, GCN, MPNN, Transformer — so we'd know immediately where we stand.

## The practical pipeline

1. **Python script** (RDKit): load SMILES strings → compute 1024-bit ECFP → save as binary vectors
2. **New C++ dataloader**: read the binary vectors (trivial — just packed bits)
3. **HCNN at DIM=10**: 1024 vertices, binary classification, AUC-ROC evaluation
4. **Compare**: vs published RF/MLP/GNN baselines

The featurization is a one-time offline step. The C++ side just sees binary vectors — same as MNIST but with natural hypercube structure instead of forced embedding.
