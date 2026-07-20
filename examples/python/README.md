# Python examples (Tier 1)

Contract-first recipes for **hypercube-cnn**. Not installed with the wheel —
clone the repo and install the package, then run from any directory.

```bash
cd HypercubeCNN
pip install .

python examples/python/synthetic_classification.py
python examples/python/synthetic_regression.py
python examples/python/arch_and_weights_io.py
```

| Script | Checks |
|--------|--------|
| `synthetic_classification.py` | Train a few epochs; accuracy improves |
| `synthetic_regression.py` | Train a few epochs; MSE drops |
| `arch_and_weights_io.py` | `export_arch` + HCNW round-trip; bit-identical logits |

Kept small and offline (no dataset download). Full API:
[docs/Python_SDK.md](../../docs/Python_SDK.md).
