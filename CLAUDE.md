# Use JAX differentiability obtain climate equilibrium

## Environment
The conda environment please use the miniconda3 that lives in `$HOME/miniconda3` with the environment `jaxesm`. Do not proceed if you cannot find this environment.

## Character
You are a very disciplined programmer, so you will keep the code flexible, easy to read, and reusable.

## Project Overview
This project is to use JAX differentiability to find climate equilibrium solution.

## Core Mandates -- strictly enforced

1. **File Permission**: Only allow writting or modifying files in this workspace (same folder where this `CLAUDE.md` lives.
2. **Implementation**: 
    - Use Python 3 for the core interpolation logic.
    - Required Python packages: `jax`, `xarray`, `argparse`, and `netCDF4` (implied for xarray).
    - Use `argparse` for handling command-line arguments in Python.
    - Python scripts must be executed via customizable bash scripts.
3. **GIT**: Do not commit without permission.

## Conventions
- Python code should follow PEP 8 standards.
- Bash scripts should be well-documented and include error handling.

## Project Structure

```
configs/
  base.py             # shared Config dataclass and defaults
  run_<name>.py       # per-experiment overrides; imports and instantiates Config from base.py
experiments/
  run_<name>/         # outputs, plots, and checkpoints for each experiment run
src/
  ...                 # core reusable model and utility code; must be parameter-agnostic
```

### Experiment config pattern
Each `configs/run_<name>.py` instantiates a `Config` from `configs/base.py` with only the fields that differ:
```python
from configs.base import Config

cfg = Config(
    learning_rate=1e-3,
    initial_temp=288.0,
    optimizer="lbfgs",
)
```
The run script accepts a config path as an argument and imports `cfg` from it.

## Files and descriptions

### `src/`
- `aquaplanet_equilibrium.py` — main experiment script; uses gradient descent (via JAX + optax) to find equilibrium SST on an aquaplanet using the `jcm` climate model
- `optimizers.py` — custom optimizer utilities; includes helper to stack dataclass objects for use with optax
- `plot.py` — plots training results (SST vs. truth) for a single optimizer run
- `plot_spinup.py` — plots spin-up diagnostics (loss and SST evolution over time)
- `plot_for_EGU2026_poster.py` — compares multiple optimizer runs (e.g. RMSProp vs. RMSPropMomentum) for poster figures
- `practice_l-bfgs.py` — scratch script for testing L-BFGS via `optax.scale_by_lbfgs` on a simple quadratic
