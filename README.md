# gridstate

**Power-system State Estimation in pure Python.**

`gridstate` reconstructs the most likely operating state of an electrical
network from SCADA telemetry and a topology model: voltage magnitudes and
angles at every bus, plus the derived quantities — branch power flows,
currents and nodal injections.

The estimator is implemented on top of `numpy` and `scipy` only — there are
**no proprietary or vendor dependencies**. Input and output cross a single,
explicit, versioned data contract (a `.npz` file), so the package is fully
self-contained and decoupled from any particular network-model format.

```
 measurements (value + σ)        ┌──────────────────────┐        V, δ  (state)
 topology, impedances, taps  ──► │  gridstate.run_se    │ ──►   P, Q, I (branches)
 (SEInput, .npz contract)        │  WLS / IPM estimator │        injections, residuals
                                 └──────────────────────┘        (SEOutput contract)
```

## Status

**Pre-alpha.** The core pipeline runs end to end:

- p.u. conversion of the working model (`gridstate.units`);
- bus-admittance assembly `Ybus` / `Yf` / `Yt` (`gridstate.ybus`);
- measurement vector `z`, weights `R`, measurement index (`gridstate.z_vector`);
- **WLS** Gauss–Newton solve — measurement function `h(E)`, Jacobian
  `H = ∂h/∂E`, step `ΔE = (HᵀR⁻¹H)⁻¹HᵀR⁻¹r` (`gridstate.algebra`,
  `gridstate.algorithms.wls`), with SHGM-IRLS robust re-weighting;
- **IPM** interior-point estimator as an alternative solver;
- result write-back (V, δ, branch P/Q, currents, injections) into the output
  contract;
- validation: χ² test (`gridstate.validation.chi2_test`), bad-data removal by
  normalized residuals / `rn_max` (`gridstate.validation.bad_data`),
  observability analysis (`gridstate.validation.observability`).

**Tests:** 445 passing on Python 3.10–3.12, dependency-free
(`numpy`/`scipy` only).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime dependencies are just `numpy>=1.24` and `scipy>=1.10`. Python 3.10+.

## Quick start

The public boundary is a `.npz` data-contract file. Load it, run the
estimator, read the results:

```python
from gridstate import load_se_input_npz, run_se

# 1. Load a state-estimation input contract (working model + precomputed plans).
se_input = load_se_input_npz("path/to/case.npz")

# 2. Run the estimator (WLS by default; production pipeline defaults applied).
out = run_se(se_input)

print(out.success, out.iterations, out.objective_value)

# 3. Read per-object results from the output contract.
bus = out.node(node_id=1001)        # {'voltage_magnitude': ..., 'voltage_angle': ..., ...}
flow = out.branch(branch_id=42)     # {'power_from_p': ..., 'current_from': ..., ...}
```

`run_se` returns an `SEOutput`: structured arrays `nodes` / `branches` /
`measurements` (id + result columns) plus convergence scalars
(`success`, `iterations`, `objective_value`, `algorithm`) and the state
vectors `v_pu` / `delta_rad`.

To estimate from a model you already hold in memory, wrap it with
`load_se_input(model)` (skips the format-dependent input stages) or call the
lower-level `estimate(model, algorithm="wls", ...)` directly.

## Data contract

Input and output are governed by an explicit, **versioned** schema
(`gridstate.contract`):

- `SE_INPUT` / `SE_OUTPUT` — table/column declarations with roles
  (`INPUT` / `WORKING` / `OUTPUT`) for `nodes`, `branches`, `measurements`,
  `generators`, and the raw side tables.
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
from gridstate import chi2_analysis, remove_bad_data, analyze_observability

report = analyze_observability(se_input.model)
assert report.is_observable, report.diagnostics

chi2 = chi2_analysis(se_input.model, chi2_prob_false=0.05)
if chi2.bad_data_present:
    cleaned = remove_bad_data(se_input.model, rn_max_threshold=3.0)
    print("removed measurements:", cleaned.removed_meas_ids)
```

## Public API

Exported from the top-level package:

| Symbol | Purpose |
| --- | --- |
| `run_se`, `run_pipeline` | run the estimation pipeline (contract / direct) |
| `estimate` | low-level single-solve entry point |
| `load_se_input_npz`, `load_se_input`, `save_se_input` | data-contract I/O |
| `SEInput`, `SEOutput`, `SEResult` | input / output / detailed result objects |
| `PipelineConfig`, `pipeline_manifest` | pipeline configuration & step manifest |
| `chi2_analysis`, `remove_bad_data`, `analyze_observability` | validation |

## Development

```bash
make install-dev    # editable install + dev/test extras + pre-commit hooks
make test           # pytest tests/
make lint           # ruff check + ruff format --check
make type-check     # mypy gridstate
make check          # format + lint + type-check + test
```

Continuous integration (lint, the 3.10–3.12 test matrix, and a build) runs on
GitHub Actions; see [`.github/workflows/`](.github/workflows/).

## License

`gridstate` is released under the **MIT License** — see [`LICENSE`](LICENSE).

The estimation core (WLS/IPM linear algebra, χ² test and bad-data detection)
is adapted from [pandapower](https://github.com/e2nIEE/pandapower), which is
distributed under the BSD 3-Clause License. The full third-party notice and
the list of adapted files are reproduced in the "Third-Party Software Notices"
section of [`LICENSE`](LICENSE).
