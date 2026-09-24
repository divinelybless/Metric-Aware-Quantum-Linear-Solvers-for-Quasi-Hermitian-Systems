# Metric-Aware Quantum Linear Solvers for Quasi-Hermitian Systems

Numerical reproducibility repository for the manuscript:

**Metric-Aware Quantum Linear Solvers for Quasi-Hermitian Systems: Conditioning, Optimal Metrics, and Exceptional-Point Obstructions**

## Contents

- `numerics_submission.py` — reproduces the numerical figures, LaTeX tables, and CSV data used in the submission manuscript.
- `requirements.txt` — Python dependencies.
- `data/pt_verification.csv` — PT-symmetric dimer verification data.
- `data/hatano_nelson_resources.csv` — shifted open-boundary Hatano–Nelson resource data.

Running the script also creates `figures/` and `tables/` directories containing the plotted numerical results and LaTeX tables.

## Reproduce the numerical results

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python numerics_submission.py
```

The submission calculations use:

- QSVT proxy tolerance: `1e-6`
- reproducibility seed: `20260924`
- double-precision NumPy/SciPy linear algebra

## Numerical models

The script reproduces the two main numerical benchmarks in the manuscript:

1. the PT-symmetric dimer, including condition-number, pseudospectral, projector, and QSVT-degree diagnostics;
2. the shifted open-boundary Hatano–Nelson chain, including singular-value scaling, effective inverse scales, coherent plain-QSVT resource proxies, and the leading logical Clifford+T accounting stated in the manuscript.

## Scope of the resource estimates

The Hatano–Nelson resource comparison is explicitly scoped to the plain-QSVT-plus-amplitude-amplification architecture and the logical Clifford+T cost model defined in the manuscript. It is not a surface-code hardware estimate and is not claimed as a lower bound against every possible structure-specialized non-Hermitian quantum solver.

## Data availability

All numerical data in this repository are generated from the equations and parameter choices reported in the manuscript. No experimental or third-party datasets are required.
