# CNHypergraph

`disease_mappings.py` contains the mappings for the ICD blocks and 34 chronic-disease groups. `FirstScript.ipynb` prepares the analytical tables, `DescriptiveAnalysis.ipynb` contains the descriptive analyses, and `Hy.ipynb` performs Hy-MMSBM analysis.

## Hy-MMSBM

The original Hy-MMSBM implementation is vendored in `third_party/Hy-MMSBM` under its MIT license.

- Upstream: https://github.com/nickruggeri/Hy-MMSBM
- Commit: `6a12077120088857c2cd90f86cf0c28a80fe1c80`
- Local modifications: none

Hy uses the Python executable of the selected notebook kernel. Create and select a Python 3.9 environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root `requirements.txt` directly lists the complete dependency set for the project notebooks and the vendored Hy-MMSBM implementation.
