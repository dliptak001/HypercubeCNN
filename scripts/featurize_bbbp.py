"""
Featurize BBBP dataset: SMILES -> 1024-bit ECFP4 fingerprints.

Downloads BBBP.csv from MoleculeNet, computes fingerprints via RDKit,
writes a binary file that the C++ MoleculeNetDataset can read.

Output format (.hcfp):
  Header (16 bytes):
    4 bytes: magic "HCFP"
    4 bytes: uint32_le num_samples
    4 bytes: uint32_le num_bits (1024)
    4 bytes: uint32_le num_tasks (1)
  Per sample (num_bits + num_tasks + 1 bytes):
    num_bits bytes: fingerprint (each byte 0 or 1)
    num_tasks bytes: labels (0 or 1; 255 = missing)
    1 byte: split (0=train, 1=val, 2=test)

Usage:
    python featurize_bbbp.py [output_dir]
    Defaults to ../data/
"""

import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
NUM_BITS = 1024
RADIUS = 2  # ECFP4 (radius 2 * 2 = diameter 4)


def download_bbbp(data_dir: Path) -> Path:
    csv_path = data_dir / "BBBP.csv"
    if csv_path.exists():
        print(f"Using cached {csv_path}")
        return csv_path
    print(f"Downloading BBBP.csv...")
    urllib.request.urlretrieve(BBBP_URL, csv_path)
    print(f"Saved to {csv_path}")
    return csv_path


def compute_fingerprint(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NUM_BITS)
    return np.array(fp, dtype=np.uint8)


def scaffold_split(smiles_list: list[str], train_frac=0.8, val_frac=0.1,
                   seed=42) -> list[int]:
    """Scaffold-based split: molecules with the same Murcko scaffold stay together."""
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
        scaffolds.setdefault(scaffold, []).append(i)

    # Sort scaffolds by size (largest first) for deterministic splitting
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)

    n = len(smiles_list)
    train_cutoff = int(n * train_frac)
    val_cutoff = int(n * (train_frac + val_frac))

    splits = [0] * n  # default train
    count = 0
    for group in scaffold_groups:
        for idx in group:
            if count < train_cutoff:
                splits[idx] = 0  # train
            elif count < val_cutoff:
                splits[idx] = 1  # val
            else:
                splits[idx] = 2  # test
            count += 1

    return splits


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_bbbp(out_dir)
    df = pd.read_csv(csv_path)

    print(f"BBBP: {len(df)} molecules")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution: {df['p_np'].value_counts().to_dict()}")

    # Compute fingerprints
    smiles_col = "smiles"
    label_col = "p_np"  # 1 = permeable, 0 = not permeable

    fingerprints = []
    labels = []
    valid_smiles = []
    skipped = 0

    for _, row in df.iterrows():
        fp = compute_fingerprint(row[smiles_col])
        if fp is None:
            skipped += 1
            continue
        fingerprints.append(fp)
        labels.append(int(row[label_col]))
        valid_smiles.append(row[smiles_col])

    print(f"Valid: {len(fingerprints)}, Skipped: {skipped}")

    # Scaffold split
    splits = scaffold_split(valid_smiles)
    train_n = sum(1 for s in splits if s == 0)
    val_n = sum(1 for s in splits if s == 1)
    test_n = sum(1 for s in splits if s == 2)
    print(f"Split: train={train_n}, val={val_n}, test={test_n}")

    # Fingerprint statistics
    fp_array = np.array(fingerprints)
    density = fp_array.mean()
    print(f"Fingerprint density: {density:.3f} ({density * NUM_BITS:.0f} bits set on average)")

    # Write binary file
    out_path = out_dir / "bbbp_ecfp4_1024.hcfp"
    num_samples = len(fingerprints)
    num_tasks = 1

    with open(out_path, "wb") as f:
        # Header
        f.write(b"HCFP")
        f.write(struct.pack("<III", num_samples, NUM_BITS, num_tasks))

        # Samples
        for i in range(num_samples):
            f.write(bytes(fingerprints[i]))       # 1024 bytes (0 or 1)
            f.write(bytes([labels[i]]))            # 1 byte
            f.write(bytes([splits[i]]))            # 1 byte

    file_size = out_path.stat().st_size
    print(f"Wrote {out_path} ({file_size:,} bytes)")
    print(f"  {num_samples} samples x ({NUM_BITS} bits + {num_tasks} labels + 1 split)")


if __name__ == "__main__":
    main()
