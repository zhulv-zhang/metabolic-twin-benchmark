# Configuration snapshot

Settings stated in the paper's Methods that are easy to miss when re-running.

## Environments
- `venv` (docs/pip_freeze_venv.txt): mask construction, semi-synthetic generation, training, evaluation.
- `dyn` (docs/pip_freeze_dyn.txt): real-data tensor preparation from mmc3.xlsx only (adds openpyxl).

## Hash-seed recipe
- Main chain (mask builds, generation, training): PYTHONHASHSEED=0.
- Reaction-supplementation step (+590 edges): PYTHONHASHSEED=621.
- Direction replayer (code/utils/direction_finalizer.py): deterministic, seed-free.

## Real-data fit (paper section 2.3)
- Proximal settings: tau = 3e-4 with w_sparse = 1e-3, and tau = 1e-3; 600 epochs each.
- Retained GE edges: 100-119 of 3,878 across the two settings.
- Penalized-only reference retains all 3,878.

## Probe arms (paper section 2.5)
- Freeze-probe and pinning commands (zf, x0, x0b, z-correction, P-pathway):
  see docs/ARM_LIST.md.

## Not re-runnable from this repository
- The exploratory rat-cohort re-analysis (paper section 2.10) and its input cohort
  definition are not part of the released pipeline (see README, Scope).
