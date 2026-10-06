# Supermirror design and calculation audit

Read [PHYSICS_AND_CODE_REVIEW.md](PHYSICS_AND_CODE_REVIEW.md) for the physics, paper/equation references, original-code findings, and validation results.

## Outputs

- `supermirrors/Ni_Ti_supermirror_m3.npy`: 300 graded Ni/Ti bilayers.
- `supermirrors/Ni_Ti_supermirror_m5.npy`: 1600 graded Ni/Ti bilayers.
- Both arrays have columns **[pair number, Ni thickness in nm, Ti thickness in nm]**, ordered from the illuminated surface to the substrate. Vacuum and semi-infinite Si are added by the solver, not stored as rows.
- `supermirrors/design_metadata.json`: parameters, units, assumptions and sampled acceptance results.
- `supermirrors/m3_response.npz`, `m5_response.npz`: k_perp, Q, m coordinate, R, T into Si, Ni/Ti capture and lossless R.
- `supermirrors/design_validation.png` and `.pdf`: response and thickness plots.
- `analysis/audit_results.json`: independent checks and fresh original-notebook execution results.
- `analysis/material_sld_check.json`: reproduction of the stated bulk SLD values using periodictable 2.1.0.

The m=3/m=5 nominal ranges have sampled minimum R=0.9401/0.8653 in the capture-only, sharp-interface model. The m=5 design does not promise 90% reflection throughout its band. See the report for fabrication and model limitations. No original notebook or input array was overwritten.

## Reproduce in the prepared environment

```bash
cd /workspace/absorption-in-mirrors
source /workspace/.venvs/absorption-in-mirrors/bin/activate
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/.cache
export IPYTHONDIR=/workspace/.onboarding/absorption-in-mirrors/ipython
export JUPYTER_RUNTIME_DIR=/workspace/.onboarding/absorption-in-mirrors/runtime
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python generate_supermirrors.py
python audit_calculations.py --execute-notebooks
```

The generator reproduces its outputs, including validation figures and checks; `--output-dir /some/path` writes elsewhere. The audit without `--execute-notebooks` runs independent checks but omits full notebook execution from the newly written results JSON. Full execution copies input data and saves executed notebooks in temporary directories whose paths are recorded in that JSON; those paths may disappear after the machine is replaced. Summary results and supplied deliverables reside in this repository.

For a separate Python environment, install the tested dependencies and register its kernel:

```bash
python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-analysis.txt
python -m ipykernel install --sys-prefix --name python3
```

Generation requires NumPy and Matplotlib; auditing also needs PyTorch and periodictable, plus the listed notebook tools for full execution. CPU execution was validated on Python 3.12.14. PyTorch is not required by the standalone `supermirror.py` solver.

## Use a new profile

```python
import numpy as np
from supermirror import KC_NI, profile_layers, solve_fields

profile = np.load('supermirrors/Ni_Ti_supermirror_m5.npy', allow_pickle=False)
d_A, rho_A2 = profile_layers(profile, high='Ni')
k_perp = np.linspace(0.001, 0.062, 121)
result = solve_fields(k_perp, d_A, rho_A2)
R = result['R']
T = result['T']
P_Ni = result['absorption'][:, 0::2].sum(axis=1)
P_Ti = result['absorption'][:, 1::2].sum(axis=1)
np.testing.assert_allclose(R + T + P_Ni + P_Ti, 1, atol=1e-8, rtol=0)
```

Here `absorption` has shape **(wavevectors, finite layers)**. This standalone return convention differs from the original notebooks' `compute_absorption`, which returns **(finite layers, wavevectors)**. All thicknesses passed to `solve_fields` are angstroms and all SLDs are angstrom^-2, rather than the notebooks' micro-SLD columns. At an exact internal zero wavevector the solver raises a clear exception; a boundary-value method with the linear-in-depth solution is required for that limit.

To reuse a profile in the original legacy notebooks, change `DATA_PATH`, explicitly set the high-SLD layers to **Ni**, and use the consistent constants below. Merely changing the path retains their original Be setup and misleading Ni labels:

```python
DATA_PATH = Path('supermirrors/Ni_Ti_supermirror_m5.npy')
# After the existing layer-array construction (SLDs in 10^-6 angstrom^-2):
layers_test[1:-1:2, 1] = 9.407764749850172
layers_test[1:-1:2, 2] = 0.0011404490352502582
layers_test[2:-1:2, 1] = -1.9248657571990726
layers_test[2:-1:2, 2] = 0.0009673155188374778
layers_test[-1, 1] = 2.0737423003838087
layers_test[-1, 2] = 0.000023757942260997373
# Extend the scan past the m=5 edge at k_perp=0.054365 angstrom^-1:
q_test = torch.linspace(0.001, 0.062, 5000, dtype=torch.float64, device=device)
```

These are suggested edits for subsequent notebook work; the audit keeps the originals as evidence. The comparison cells 11–12 separately overwrite material constants and would also need harmonization for a consistent material comparison.

The configurable notebook accepts named layers instead of profile arrays. In its settings cell, a new stack can be constructed as follows:

```python
profile = np.load('supermirrors/Ni_Ti_supermirror_m3.npy', allow_pickle=False)
stack = [('air', 0)]
for _, d_ni_nm, d_ti_nm in profile:
    stack.extend([('Ni', 10*d_ni_nm), ('Ti', 10*d_ti_nm)])
stack.append(('Si', 0))
STRUCTURES = [stack]
K_MAX = 0.038  # includes the m=3 edge and roll-off; use >=0.06 for m=5
```

For thousands of layers, avoid allocating a very fine full-depth heatmap by default: its size scales with wavevector count times depth-grid count. The supplied generator computes response curves in batches and does not require a depth grid for analytic capture. The configurable notebook's default 0.5 angstrom depth spacing is expensive for these coatings.
