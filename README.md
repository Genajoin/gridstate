# gridstate

Power-system State Estimation in pure Python.

`gridstate` reconstructs the most likely operating state of an electrical
network from SCADA telemetry and a topology model: voltage magnitudes and
angles at every bus, plus the derived quantities — branch power flows,
currents and nodal injections.

The estimator is implemented on top of `numpy` and `scipy` only — there are
no proprietary or vendor dependencies. Input and output cross a single,
explicit, versioned data contract (a `.npz` file), so the package is fully
self-contained and decoupled from any particular network-model format.

```
 measurements (value + σ)        ┌──────────────────────┐        V, δ  (state)
 topology, impedances, taps  ──► │  gridstate.run_se    │ ──►   P, Q, I (branches)
 (SEInput, .npz contract)        │  WLS / IPM estimator │        injections, residuals
                                 └──────────────────────┘        (SEOutput contract)
```

## Status

Early-stage, actively developed. Releases follow
[Semantic Versioning](https://semver.org/); the input/output data contract has
its own version (`gridstate.contract.CONTRACT_VERSION`). Breaking changes are
listed in the [changelog](CHANGELOG.md).

What the library provides:

- **Estimators.** WLS Gauss–Newton (`gridstate.algorithms.wls`) with SHGM-IRLS
  robust re-weighting, and an interior-point method (IPM,
  `gridstate.algorithms.ipm`) with box constraints on load / generation and
  soft nodal balance rows. Measurement function `h(E)`, Jacobian `H = ∂h/∂E`
  and the normal-equation step live in `gridstate.algebra`.
- **Pipeline.** `gridstate.pipeline` chains the preparation of a network and
  its measurements (p.u. conversion, `Ybus`, topology and tap handling,
  measurement sanity filters, generator aggregation, node-type refinement),
  the solve, and optional post-solve passes. Every step has a toggle and every
  parameter has declared bounds (see [Pipeline configuration](#pipeline-configuration)).
- **Pseudo-measurements.** Unobserved quantities get weak priors (zero or
  materialized injections, flat voltage placeholders) so that partially
  observable models still solve; priors are flagged and kept apart from
  telemetry in all reports.
- **Voltage refinement.** `v_refine` tightens consistent voltage measurements;
  `v_mirror` lifts the voltage placeholders of blind clusters to the measured
  boundary (with a cross-transformer fallback and a chain variant for deep
  radial chains); `refine_merge` runs the refine family with a single warm
  re-solve.
- **Bad-data handling.** A bad-data re-pass on the residuals of the solved
  estimate flips sign-reversed channels and rejects dead or conflicting ones;
  `gridstate.validation` adds the χ² test, the largest-normalized-residual
  removal (critical measurements are never removed) and observability analysis.
- **Revert policy.** A post-solve pass that edits measurements and re-solves is
  reverted — measurements and result restored — if the previous solution
  converged and the re-solve did not; the step statistics then carry
  `reverted=True` and the reason.
- **Quality summary.** χ² summary, worst normalized residuals (telemetry only
  by default, `scope="all"` includes priors) and worst nodal imbalances, in
  `SEResult`; `populate_quality_summary` fills it for a result you hold.
- **Losses.** `compute_system_losses` splits active and reactive losses into
  series, charging and shunt parts per branch and node, in the standard sign
  convention (capacitive elements give negative reactive "losses").
- **Regression harness.** A golden snapshot harness (`tools/golden.py`,
  `tests/_golden.py`) records every output array and the event sequence of a
  matrix of cases and configurations, so refactoring and performance changes
  can be proven result-identical.

The test suite runs on Python 3.10–3.12 in continuous integration and needs
only `numpy` and `scipy`; the optional extras are exercised by a dedicated job.

## Installation

```bash
pip install gridstate
```

or, for development, from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

Runtime dependencies are just `numpy>=1.24` and `scipy>=1.10`. Python 3.10+.

### Extras

| Extra | Adds | Purpose |
| --- | --- | --- |
| `fast` | `cvxopt` | CHOLMOD (SuiteSparse) Newton solves with symbolic factorization reuse; without it the solver falls back to `scipy.sparse.linalg.spsolve` (`kkt_solver="auto"` picks the best available backend) |
| `numba` | `numba` | IPM assembles `HᵀR⁻¹H` with a numba kernel on the fixed Jacobian pattern instead of scipy sparse products; enables the opt-in `selected_inversion` method of the normalized residuals |
| `test-models` | `pandapower` | regenerating the IEEE test fixtures with `tools/gen_ieee_fixtures.py`; not needed to run the tests |
| `test`, `dev` | pytest, ruff, mypy, build tools, … | running and developing the package |
| `build` | Cython and build tooling | the optional compiled distribution (`build_cython.py`) |

`fast` and `numba` speed up large models without changing any result bit. Use
them together: the numba kernel feeds the CHOLMOD backend.

```bash
pip install "gridstate[fast,numba]"
```

Notes on the numba path:

- The kernel is compiled on first use in a process (about 0.5 s) and cached on
  disk next to the package, or in numba's user cache when that is not writable.
  Point `NUMBA_CACHE_DIR` at a writable directory to control the location (for
  example a read-only container image).
- Long-running services can warm it up with
  `gridstate.algorithms.kkt_solver.numba_gram_available()`.
- `GRIDSTATE_NUMBA=0` switches the numba path off; the scipy path is used
  instead and gives identical results.
- The quality summary can compute `diag(H G⁻¹ Hᵀ)` of the normalized
  residuals by selected inversion of the sparse Cholesky factor
  (`PipelineConfig.normalized_residuals_method="selected_inversion"`, or
  `estimate(..., quality_summary_rn_method="selected_inversion")`): many times
  faster than the default per-measurement solves on large models. It is
  opt-in because it is not bit-identical: the two methods agree to rounding,
  except on nearly critical measurements (`Ω_ii/σ²_i` tiny), which the
  quality summary lists in `near_critical_measurements` with either method.
  Without numba (or with `GRIDSTATE_NUMBA=0`) the default method is used.

## Quick start

The public boundary is a `.npz` data-contract file. Load it, run the
estimator, read the results (the repository ships IEEE 14/30/118-bus cases in
`tests/test_data/ieee/`):

```python
from gridstate import load_se_input_npz, run_se

# 1. Load a state-estimation input contract (working model + precomputed plans).
se_input = load_se_input_npz("tests/test_data/ieee/case14.npz")

# 2. Run the estimator (WLS by default; pipeline defaults applied).
out = run_se(se_input)

print(out.success, out.iterations, out.objective_value)

# 3. Read per-object results from the output contract.
bus = out.node(node_id=int(out.nodes["id"][0]))
# {'id': ..., 'voltage_magnitude': ..., 'voltage_angle': ..., 'p_inj_calc': ..., ...}
flow = out.branch(branch_id=int(out.branches["id"][0]))
# {'id': ..., 'power_from_p': ..., 'current_from': ..., 'loss_p': ..., ...}
```

`run_se` returns an `SEOutput`: structured arrays `nodes` / `branches` /
`measurements` (id + result columns) plus convergence scalars (`success`,
`iterations`, `objective_value`, `algorithm`, `convergence_status`) and the
state vectors `v_pu` / `delta_rad`.

To estimate from a model you already hold in memory, wrap it with
`load_se_input(model)` (skips the format-dependent input stages), call
`run_pipeline(model, config=...)` for the full pipeline and a detailed
`SEResult`, or call the lower-level `estimate(model, algorithm="wls", ...)`
directly for a single solve.

## Pipeline configuration

`run_se` and `run_pipeline` take a `PipelineConfig`: a dataclass whose fields
select the algorithm and its options, switch the optional steps on or off, and
set their thresholds. Values are validated against the declared metadata
(choices, bounds, NaN, wrong types) when the config is created and again when a
run starts; all problems are reported in one `ValueError`.

```python
from gridstate import PipelineConfig, load_se_input_npz, run_se

cfg = PipelineConfig(
    algorithm="ipm",   # "wls" (default) or "ipm"
    bad_data=True,     # bad-data re-pass on the solved estimate
    v_refine=True,     # tighten consistent voltage measurements
    v_mirror=True,     # lift blind-cluster voltage placeholders
)
out = run_se(load_se_input_npz("tests/test_data/ieee/case14.npz"), config=cfg)
print(out.convergence_status, out.success)
```

`pipeline_manifest()` returns a JSON-serializable description of the pipeline —
the steps in execution order (with their toggles and default state), the
parameters (type, default, bounds, choices, help) and the section order — so
that a UI can build its form without hard-coding any default. Pass
`config=` to describe a downstream configuration instead of the library
defaults: a `PipelineConfig` subclass (instantiated without arguments) or an
instance.

```python
from dataclasses import dataclass

from gridstate import PipelineConfig, pipeline_manifest


@dataclass
class MyConfig(PipelineConfig):
    algorithm: str = "ipm"
    bad_data: bool = True


manifest = pipeline_manifest(config=MyConfig)
defaults = {p["name"]: p["default"] for p in manifest["params"]}
assert defaults["algorithm"] == "ipm"
```

## Data contract

Input and output are governed by an explicit, versioned schema
(`gridstate.contract`):

- `SE_INPUT` / `SE_OUTPUT` — table/column declarations with roles
  (`INPUT` / `WORKING` / `OUTPUT`) for `nodes`, `branches`, `measurements`,
  `generators`, and the side tables (`tap_steps`, `load_characteristics`,
  `shunts`).
- `CONTRACT_VERSION` follows SemVer; `is_data_compatible(...)` gates whether a
  given `.npz` can be consumed.
- `load_se_input_npz` / `save_se_input` round-trip the contract to disk.
- `validate_input(...)` checks a model against the contract and fails early on
  missing required columns or an incompatible version.

This contract is what lets `gridstate` stand alone: the estimator never
depends on a concrete external network-model class — only on the data it
declares it needs.

## Validation

```python
from gridstate import analyze_observability, chi2_analysis, load_se_input_npz, remove_bad_data

se_input = load_se_input_npz("tests/test_data/ieee/case14.npz")

report = analyze_observability(se_input.model)
assert report.is_observable, report.diagnostics

chi2 = chi2_analysis(se_input.model, chi2_prob_false=0.05)
if chi2.bad_data_present:
    cleaned = remove_bad_data(se_input.model, rn_max_threshold=3.0)
    print("removed measurements:", cleaned.removed_meas_ids)
```

## Public API

Exported from the top-level package (`gridstate.__all__`):

| Symbol | Purpose |
| --- | --- |
| `run_se` | run the pipeline on a contract input, returning an `SEOutput` |
| `run_pipeline` | run the pipeline on a model in memory, returning an `SEResult` |
| `estimate` | low-level single-solve entry point (WLS or IPM) |
| `load_se_input_npz`, `load_se_input`, `save_se_input` | data-contract I/O |
| `SEInput`, `SEOutput`, `SEResult` | input / output / detailed result objects |
| `Chi2Summary`, `ResidualRow`, `ImbalanceRow` | quality summary rows carried by `SEResult` |
| `populate_quality_summary` | fill the quality summary of a result you hold |
| `PipelineConfig`, `pipeline_manifest` | pipeline configuration and its step / parameter manifest |
| `chi2_analysis`, `Chi2Result` | χ² test of the estimate |
| `remove_bad_data`, `BadDataResult` | largest-normalized-residual bad-data removal |
| `compute_normalized_residuals_report`, `NormalizedResidualReport` | normalized residuals of all measurements |
| `analyze_observability`, `ObservabilityReport` | observability analysis |
| `compute_system_losses`, `SystemLosses`, `BranchLossRow` | active / reactive loss breakdown |
| `__version__` | installed package version |

## Development

```bash
make install-dev    # editable install + dev/test extras + pre-commit hooks
make test           # pytest tests/
make lint           # ruff check + ruff format --check
make type-check     # mypy gridstate
make check          # lint + type-check + test (does not modify files)
make format         # apply ruff formatting and autofixes
```

Result-preserving changes are checked with the golden harness (same machine,
bit-exact):

```bash
python tools/golden.py record --out /tmp/golden-base      # on the base revision
python tools/golden.py compare --reference /tmp/golden-base --strict
```

`tests/test_neutral_wording.py` keeps the library text self-contained (no
references to external programs, input formats or deployments).

Continuous integration (lint, mypy, the 3.10–3.12 test matrix with branch
coverage, a CHOLMOD + numba job, and a build with a wheel smoke test) runs on
GitHub Actions; see [`.github/workflows/`](.github/workflows/).

## License

`gridstate` is released under the MIT License — see [`LICENSE`](LICENSE).

The estimation core (WLS/IPM linear algebra, χ² test and bad-data detection)
is adapted from [pandapower](https://github.com/e2nIEE/pandapower), which is
distributed under the BSD 3-Clause License. The full third-party notice and
the list of adapted files are reproduced in the "Third-Party Software Notices"
section of [`LICENSE`](LICENSE).
