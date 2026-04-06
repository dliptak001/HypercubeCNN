"""
Featurize MoleculeNet datasets: SMILES -> 1024-bit ECFP4 fingerprints.

Downloads CSVs from MoleculeNet, computes fingerprints via RDKit,
scaffold-splits, writes .hcfp binary files for the C++ dataloader.

Usage:
    python featurize_molnet.py [dataset] [output_dir]
    dataset: bbbp, tox21, hiv, bace, or "all" (default: all)
    output_dir: defaults to ../data/
"""

import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

NUM_BITS = 1024
RADIUS = 2  # ECFP4

# Dataset definitions: url, smiles_col, label_cols, compression
DATASETS = {
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "file": "BBBP.csv",
        "smiles": "smiles",
        "labels": ["p_np"],
        "compression": None,
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "file": "tox21.csv.gz",
        "smiles": "smiles",
        "labels": ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
                    "SR-HSE", "SR-MMP", "SR-p53"],
        "compression": "gzip",
    },
    "hiv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "file": "HIV.csv",
        "smiles": "smiles",
        "labels": ["HIV_active"],
        "compression": None,
    },
    "bace": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "file": "bace.csv",
        "smiles": "mol",
        "labels": ["Class"],
        "compression": None,
    },
}


def download(url: str, path: Path) -> Path:
    if path.exists():
        print(f"  Using cached {path}")
        return path
    print(f"  Downloading {path.name}...")
    urllib.request.urlretrieve(url, path)
    return path


def compute_fingerprint(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NUM_BITS)
    return np.array(fp, dtype=np.uint8)


def scaffold_split(smiles_list: list[str], train_frac=0.8, val_frac=0.1,
                   seed=42) -> list[int]:
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
        scaffolds.setdefault(scaffold, []).append(i)

    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)

    n = len(smiles_list)
    train_cutoff = int(n * train_frac)
    val_cutoff = int(n * (train_frac + val_frac))

    splits = [0] * n
    count = 0
    for group in scaffold_groups:
        for idx in group:
            if count < train_cutoff:
                splits[idx] = 0
            elif count < val_cutoff:
                splits[idx] = 1
            else:
                splits[idx] = 2
            count += 1
    return splits


def featurize_dataset(name: str, data_dir: Path):
    cfg = DATASETS[name]
    print(f"\n=== {name.upper()} ===")

    csv_path = download(cfg["url"], data_dir / cfg["file"])
    df = pd.read_csv(csv_path, compression=cfg["compression"])
    print(f"  {len(df)} molecules, columns: {list(df.columns)[:8]}...")

    label_cols = cfg["labels"]
    num_tasks = len(label_cols)

    fingerprints = []
    labels_all = []  # list of lists, one per sample
    valid_smiles = []
    skipped = 0

    for _, row in df.iterrows():
        fp = compute_fingerprint(str(row[cfg["smiles"]]))
        if fp is None:
            skipped += 1
            continue

        sample_labels = []
        for col in label_cols:
            val = row[col]
            if pd.isna(val):
                sample_labels.append(255)  # missing
            else:
                sample_labels.append(int(val))
        fingerprints.append(fp)
        labels_all.append(sample_labels)
        valid_smiles.append(str(row[cfg["smiles"]]))

    print(f"  Valid: {len(fingerprints)}, Skipped: {skipped}")

    # Label stats for first task
    first_labels = [l[0] for l in labels_all if l[0] != 255]
    pos = sum(1 for l in first_labels if l == 1)
    print(f"  {label_cols[0]}: {pos}/{len(first_labels)} positive ({100*pos/len(first_labels):.0f}%)")
    if num_tasks > 1:
        missing_rates = []
        for t in range(num_tasks):
            missing = sum(1 for l in labels_all if l[t] == 255)
            missing_rates.append(f"{label_cols[t]}:{100*missing/len(labels_all):.0f}%")
        print(f"  Missing rates: {', '.join(missing_rates)}")

    # Scaffold split
    splits = scaffold_split(valid_smiles)
    train_n = sum(1 for s in splits if s == 0)
    val_n = sum(1 for s in splits if s == 1)
    test_n = sum(1 for s in splits if s == 2)
    print(f"  Split: train={train_n}, val={val_n}, test={test_n}")

    # Fingerprint density
    density = np.array(fingerprints).mean()
    print(f"  FP density: {density:.3f} ({density * NUM_BITS:.0f} bits avg)")

    # Write .hcfp
    out_path = data_dir / f"{name}_ecfp4_1024.hcfp"
    num_samples = len(fingerprints)

    with open(out_path, "wb") as f:
        f.write(b"HCFP")
        f.write(struct.pack("<III", num_samples, NUM_BITS, num_tasks))
        for i in range(num_samples):
            f.write(bytes(fingerprints[i]))
            f.write(bytes(labels_all[i]))
            f.write(bytes([splits[i]]))

    print(f"  Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if which == "all":
        for name in DATASETS:
            featurize_dataset(name, out_dir)
    elif which in DATASETS:
        featurize_dataset(which, out_dir)
    else:
        print(f"Unknown dataset: {which}")
        print(f"Available: {', '.join(DATASETS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
