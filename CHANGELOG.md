# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries follow the [Conventional Commits](https://www.conventionalcommits.org/)
types of the merged pull requests; breaking and behavior-changing items are
listed separately under "Breaking changes" and "Behavior changes". A release is
cut by tagging the merged main commit; the version is derived from that tag.

## Unreleased

### Breaking changes

- `compute_system_losses`: the reactive power of node shunts and line charging
  is reported in the standard sign convention (Y = g + jb, b > 0 capacitive).
  `q_node_shunt`, `q_branch_shunt`, per-branch `q_loss_shunt` / `q_loss_total`
  and `total_q_loss_mvar` change sign: capacitive elements now give negative
  reactive "losses", reactors positive. State estimation results are not
  affected (#40)
- Removed code without a production path (#46):
  - pipeline steps `synthesize_injections` and `resolve_merged` with their
    `PipelineConfig` fields, and `PipelineConfig.ipm_transit_balance_sigma2_pu`
    / `ipm_bound_relax`;
  - modules `preprocessing/synth_injection.py`, `preprocessing/chain_voltage.py`,
    `preprocessing/dead_gen_nodes.py`, `preprocessing/one_sided.py` and the
    `telemetry/topology.py` re-export facade;
  - `extract_boundary_node_ids_from_model`, `pipeline.default_config`,
    `detect_nominal_substitution`, the flat `q_inconsistency_*` fields of
    `TelemetryApplyConfig`, and the boundary / block-bus / terminal /
    voltage-prior options of `add_pseudo_measurements` (`PseudoMeasConfig` is
    trimmed accordingly);
  - the `build_ipm_setup(transit_balance_sigma2_pu, prior_sigma2_inj_pu)` and
    `refine_node_types_from_generators(node_load_props)` parameters;
  - the always-zero step statistics `boundary_nodes`, `block_buses`,
    `block_bus_v_loose`, `downweighted_nominal_substitution`.
- `PipelineConfig` is validated against its field metadata (choices, `min` /
  `max`, NaN, non-numbers including `bool`, `None` where not allowed) at
  construction and again at `run` / `prepare_network`, with all problems in one
  `ValueError`. Invalid configs that used to fail late, or not at all, now fail
  early: `algorithm="IPM"` is rejected at construction, and negative
  `bad_data_sigma_cap` / `flow_sigma_floor_kv_frac` are rejected (#50)
- `gridstate.pipeline` is now a package. It re-exports every name of the former
  module (including private ones) except `estimate` and `classify_v_mirror`;
  patch `gridstate.pipeline.passes` instead of `gridstate.pipeline` in tests (#50)
- Invariants that were `assert` statements now raise `ValueError` (missing
  derived plan, naming it) or `RuntimeError` (no solution), also under
  `python -O` (#50)

### Behavior changes

- New defaults of three `PipelineConfig` options, evaluated on 8
  production-size inputs; the previous behaviour stays available by setting
  the old value explicitly. Results of IPM runs, of warm starts on inputs with
  pseudo-V priors, and the last digits of `r_N` change:
  - `ipm_trust_region_scope="state"` (was `"all"`; also the default of
    `estimate(..., trust_region_scope=)` and `solve_ipm(tr_exclude_box=True)`):
    inner Newton iterations -7…-67 %, wall time ×0.47–0.95, estimate quality
    equal to the third digit (V shift ≤ 0.017 kV) (#59)
  - `warm_start_mode="x0_only"` (was `"seed_inputs"`): a warm-started result
    matches the cold one to the solver tolerance (≤ 1.4e-3 kV, against
    0.49–2.31 kV and a drift up to 4.37 kV over a 4-run chain with
    `"seed_inputs"`) (#58)
  - `normalized_residuals_method="selected_inversion"` with
    `selected_inversion_refine_below=1e-2` (was `"solve"`; also the default of
    `estimate(quality_summary_rn_method=)`, `populate_quality_summary(rn_method=)`,
    `compute_normalized_residuals(method=)`, `compute_normalized_residuals_report`
    and `remove_bad_data`): `r_N` in 8–36 % of the time, refined rows equal to
    `"solve"`, the others within 5.6e-6 relative, identical `inf` sets and
    top-20 lists; without numba (or with `GRIDSTATE_NUMBA=0`) `"solve"` is used
    (#60)
- Duplicate node measurements are no longer merged on inputs without a
  telemetry plan (the removed step summed P/Q duplicates) (#46)
- Measurements with a non-finite value or variance are skipped like
  non-positive-variance ones instead of failing the whole estimate. Infinite
  variance used to enter with zero weight and is now excluded, which reduces the
  χ² degrees of freedom by one per such measurement (#43)
- `remove_bad_data` considers only finite normalized residuals, so critical
  measurements (r_N = inf) are never removed; they are listed in
  `BadDataResult.critical_meas_ids`, and `rn_max_history` no longer contains
  inf (#43)
- Post-solve re-solve passes (bad-data re-pass, V refine, V mirror, chain
  mirror, merged refine, anti-overshoot) restore the previous solution when the
  re-solve loses convergence; the step stats carry `reverted` and
  `revert_reason`. Anti-overshoot accepts a refined solution only if it
  converged, and its `resolve` callback must return an object with `.success`
  (#44)
- The residual computed at the accepted Armijo trial point of the IPM is
  reused, so `residual_fn` must be pure (#48)
- Manifest labels and help texts of several parameters and the `step_skipped`
  reason `no derived inputs` were reworded; field and step names are unchanged
  (#47)
- Private helpers `quality_summary._normalized_residuals`, `_hgh_diag_cholmod`
  and `_hgh_diag_dense` moved to `gridstate.algebra.normalized_residuals` as
  public names (#43)
- Report: nearly critical measurements (`Ω_ii / σ²_i` below the
  near-critical threshold, default `1e-4`) are left out of the residual
  rankings and listed apart, since their `r_N` is numerically unreliable and a
  few of them could fill the whole top list. `SEResult.worst_residuals` no
  longer contains the rows of `SEResult.near_critical_measurements`;
  `compute_normalized_residuals_report` (new argument
  `near_critical_threshold`) leaves them out of `top_records`,
  `n_above_threshold` and `rn_max_finite` and lists them in the new
  `near_critical_meas_ids` (with the ratio of every row in `omega_ratio`).
  `NearCriticalRow` gains `object_kind` / `object_id` / `branch_side` as in
  `ResidualRow`. A threshold of `0` restores the previous rankings exactly.
  The `r_N` values and the bad-data test (`remove_bad_data`) are unchanged
  (#60)
- The pipeline treats a solved state that is out of range (non-finite V/δ,
  `V/Vnom` below 1e-3 or `|δ|` beyond 2π) as unusable: a main estimate that
  leaves one ends with `success=False` and a message naming the nodes, the
  re-solve passes and the quality summary are skipped for it, and a re-solve
  pass that produces one is reverted (#90)

### Feat

- `pipeline_manifest(config=...)` accepts a `PipelineConfig` subclass or
  instance and reports its defaults, fields and toggles instead of the library
  ones; a subclass field redeclared with a bare default keeps the base UI
  metadata. Without arguments the output is unchanged (#39)
- `PipelineConfig.gen_v_calibration_mode`: `"absolute"` (default, unchanged) or
  `"tighten"` (`variance = min(current, sigma2)`, never loosens) for the
  voltage variance calibration of generator and slack nodes (#41)
- Step-ordering invariants: `Step.after` declares dependencies and the new
  public `check_step_order` verifies them at import (the dependency exists and
  precedes, network steps depend only on network steps, names are unique,
  toggles are config fields) (#50)
- `IdAllocator` with a collision check (`MeasurementIdCollisionError`) for the
  ids of added measurements (#51)
- Explicit initial point: `estimate(..., x0=InitialPoint(...))` starts the
  solver from the given node V/δ without touching the node table (and thus the
  pseudo-measurements built from it); `InitialPoint.from_result` takes it from
  a previous result, and `run(init_state=...)` also accepts an `InitialPoint`.
  `PipelineConfig.warm_start_mode`: `"seed_inputs"` (the previous behaviour:
  the previous V/δ are seeded into the node table before preprocessing, so
  they are also the operating point of the pseudo-V priors) or `"x0_only"`
  (the previous solution is only the initial point of the main solve; the
  estimation problem is the cold-start one; the default, see "Behavior
  changes") (#58)
- `PipelineConfig.ipm_trust_region_scope` (`estimate(..., trust_region_scope=)`,
  `solve_ipm(tr_exclude_box=)`): `"all"` (the previous behaviour, bit for bit) or
  `"state"` — the IPM trust region is measured on V/δ only, and the resulting
  factor scales the whole Newton step, so a generation/load box variable that
  has to move far no longer shrinks the V/δ step; boxes stay protected by
  fraction-to-boundary. `"state"` is the default, see "Behavior changes" (#59)
- Near-critical report of the quality summary: `SEResult.near_critical_measurements`
  lists (`NearCriticalRow`) the measurements whose `Ω_ii / σ²_i` is below
  `PipelineConfig.near_critical_threshold` (default `1e-4`; also
  `estimate(quality_summary_near_critical=...)`), where `Ω_ii` is a small
  difference of nearly equal numbers and `r_N` is numerically unreliable.
  `compute_normalized_residuals` returns that ratio with `r_N`. Reported only:
  `r_N` values are unchanged (#60)
- `SolveStatus`: a solver-independent termination status (`kkt`,
  `mu_schedule_done`, `tr_collapsed`, `max_iter`, `stalled`, `diverged`,
  `error`) reported by `WLSResult` / `IPMResult` / `SEResult` as
  `solve_status` and in the `estimate` and re-solve step stats, next to the
  unchanged legacy `success` / `convergence_status`. It separates a WLS step
  forced below the tolerance by a collapsed trust region from a stationary
  point, and an IPM barrier schedule whose final inner solve converged from
  one that exhausted its budget. `SEResult.objective_wls` is the plain WLS
  objective of the measurement vector at the final state (no Huber weights,
  no ½, no IPM balance/prior rows), comparable between the solvers;
  `objective_value` keeps the solver-internal definition. Additive: solver
  output, events and golden snapshots are unchanged apart from the new keys
  (#57)
- `PipelineConfig.success_policy` / `estimate(success_policy=...)`:
  `"legacy"` (default, unchanged) or `"status"` — `success` is derived from
  `solve_status` (`kkt` and `mu_schedule_done` only), so the post-solve
  passes skip a non-converged base solution and the re-solve revert (#44)
  also acts on IPM re-solves (#57)
- `normalized_residuals_method="selected_inversion"` is a hybrid: after the
  selected inverse, every measurement whose `Ω_ii / σ²_i` is below
  `PipelineConfig.selected_inversion_refine_below` (default `1e-2`; also
  `estimate(quality_summary_rn_refine_below=...)`,
  `populate_quality_summary(rn_refine_below=...)`,
  `compute_normalized_residuals(refine_below=...)`), and every measurement at
  the critical cut-off `Ω_ii ≤ 1e-12`, is recomputed by the per-row solves of
  `"solve"` on the same factorization (the dense fallback without cvxopt). On
  an ill-conditioned gain (cond ~1e14) the plain selected inverse was off by
  tens of per cent in `r_N` on near-critical measurements and could mark a
  measurement critical (`inf`) that is not; those rows now carry the
  `"solve"` values and every `inf` comes from a solve. `0` recomputes only the
  critical rows. The cost grows with the share of recomputed rows (about that
  share of the `"solve"` time); `"solve"` is unchanged bit for bit (#60)

### Fix

- Bad-data robustness: critical measurements are never removed, the
  normalized-residual test uses sparse algebra instead of dense m x m matrices,
  and non-finite measurements no longer fail the estimate (#43)
- Add the `py.typed` marker declared in package-data, so wheels advertise the
  inline types (#42)
- `compute_system_losses`: transformers are identified by the contract column
  `branch_type`, so the line / transformer split of the series losses
  (`p/q_line_series`, `p/q_trafo_series`, per-branch `tip`) is correct. The
  former guess from the p.u. parameters missed transformers with a nominal tap
  and counted every line whose charging is split between the two sides as a
  transformer. The guess is kept only for branch tables without `branch_type`,
  and no longer treats split line charging as a one-sided shunt. Totals are
  unchanged (#55)
- IEEE fixture generator: nodal P/Q injection measurements no longer include
  the power of bus shunts, which gridstate models in the Y-bus (they were
  biased by up to 21 Mvar on case14 and 40 Mvar on case118, i.e. 10-20 sigma).
  The shared helper `gridstate.adapters.network_injection` returns the
  shunt-free injection of a solved pandapower network. Fixtures and golden
  snapshots are regenerated; the WLS objective of every IEEE fixture is now
  within the chi-square bound of its noise, and the IEEE test tolerances are
  tightened to 0.008 p.u. / 1 degree (#79)
- `from_pandapower` keeps the transformer magnetising branch (`i0_percent`,
  `pfe_kw`) in the branch shunt fields, using pandapower's T equivalent circuit
  (its power-flow default) or the pi circuit, selected by the new
  `trafo_model` argument or taken from the last power flow. The series
  impedance follows pandapower's equations as well, which also fixes
  transformers with an off-nominal LV-side tap or rated voltages that differ
  from the bus voltages. The Y-bus matches pandapower to 1e-13 p.u. on IEEE
  case14 / case30 / case118 (case118 differed by 0.4 p.u.) (#80)
- A re-solve pass (anti-overshoot, bad-data re-pass, V refine / mirror) no
  longer accepts a re-solve whose state is out of range (e.g. node angles
  beyond ±2π); the previous solution is restored with `revert_reason` and
  `state_violation` in the step stats. On gross measurement errors the
  accepted state used to make the final quality summary raise `ValueError` in
  the p.u. conversion (#90)

### Perf

- Bit-identical hot-path improvements: reuse of the Armijo residual in the IPM,
  Jacobian pattern grouping with one stable argsort, an id-to-row index and
  zero-copy table views in `Working` (about -15 % wall time on the largest
  benchmark) (#48)
- The IPM Gram matrix `HᵀR⁻¹H` is assembled on the fixed Jacobian pattern with
  a numba kernel that reproduces the scipy summation order (checked bit-for-bit
  at load time, with automatic fallback). Optional `numba` extra,
  `NUMBA_CACHE_DIR`, `GRIDSTATE_NUMBA=0` and
  `kkt_solver.numba_gram_available()` for warm-up (about -36 % wall time on the
  largest benchmark) (#49)
- `PipelineConfig.normalized_residuals_method="selected_inversion"`
  (also `estimate(quality_summary_rn_method=...)`; the default, see "Behavior
  changes"): `diag(H G⁻¹ Hᵀ)` of the
  normalized residuals from the selected inverse of the sparse Cholesky factor
  (Takahashi recurrence; CHOLMOD factor, or scipy SuperLU without cvxopt)
  instead of one triangular solve per measurement. Needs the `numba` extra
  (otherwise, and with `GRIDSTATE_NUMBA=0`, `"solve"` is used).
  The quality summary of a 3000-bus synthetic network drops from 1.4 s to
  0.07 s; `r_N` agrees to ~1e-12 except on near-critical measurements.
  `"solve"` stays bit-identical (#60)

### Refactor

- Split `pipeline` into a package: `config`, `context`, `preparation`,
  `passes`, `steps`, `runner`, `manifest` (#50)
- Shared array helpers (`gridstate/_arrays.py`), one id allocator
  (`measurement_ids.py`), enum codes in place of literals, a single measurement
  noise model (`telemetry/noise.py`) and shared box predicates in `bounds.py`;
  all bit-identical (#51)
- Runtime facade `gridstate/runtime.py`; shunt-sanity orchestration in
  `pipeline/orchestration.py`; `gridstate.contract.runtime` stays as a
  compatibility shim. A single accepted shunt candidate reuses its trial
  result (3 runs instead of 4) (#70)
- One `ResolvePass` class (guard, lazy snapshot, apply, re-solve, settle or
  revert) for the refine family, declared once as `RefineStage`; sequential and
  merged refine share one code path (#70)
- Remove dead code (-4762 / +253 lines), see "Breaking changes" (#46)

### Test

- Golden regression harness (`tools/golden.py`, `tests/_golden.py`) over the
  IEEE cases and degraded variants, and a CI job that runs the suite on the
  CHOLMOD backend (#45)
- Vocabulary guard `tests/test_neutral_wording.py` keeps the library text
  self-contained; test modules renamed after their subject (#47)

### Docs and tooling

- README: current feature list, extras, pipeline configuration and a public API
  table matching `gridstate.__all__`; changelog backfilled for every release
- `__version__` prefers the setuptools_scm generated `gridstate/_version.py`
  and falls back to the package metadata, so an editable install with stale
  metadata no longer reports an old version
- Release comments and `[tool.commitizen]` no longer suggest `cz bump` or a
  changelog update on bump; mypy is pinned and targets Python 3.10; branch
  coverage is enabled; `make check` no longer modifies files
- `build_cython.py` leaves the PyInstaller hook and the numba kernel module as
  plain Python and documents how it is used

## v1.13.0 (2026-09-30)

### Perf

- Vectorized assembly of the measurement vector (#38)
- Fixed-pattern measurement Jacobian (bit-identical to the generic sparse
  path, which remains the fallback), sparse CHOLMOD evaluation of
  `diag(H G⁻¹ Hᵀ)` for the quality summary, and early stop of anti-overshoot
  rounds that did not lower the maximum V/Vnom (#37)

## v1.12.0 (2026-09-29)

### Feat

- `ipm_boxed_balance_weight_factor` (default 0, off): nodes whose admissible
  injection is fully set by declared load / generation boxes get a tighter
  balance row, so a conflict between measurements shows up as a residual
  rather than as a node imbalance (#36)

## v1.11.2 (2026-09-28)

### Feat

- `loosen_unknown_load_pseudo` (default on): the zero injection prior of a
  node with load but no load data is loosened where the incident flow (P) or
  flow / voltage (Q) measurements determine the injection anyway (#35)

### Fix

- Drop generation-only P/Q injections on nodes whose load is unknown (unset
  box); they used to pin the load to zero. New `FilterFlag` value
  `GENERATION_ONLY_INJECTION`, with `released_p` / `released_q` in the step
  result (#34)

## v1.11.1 (2026-09-28)

### Fix

- Generator-node voltage calibration ignores the sentinel generation bounds and
  no longer overwrites the variance of questionable readings; only readings of
  good quality are tightened (#33)

## v1.11.0 (2026-09-28)

### Feat

- `relax_generation_min_to_injection` (default off): relax a minimum
  generation that the node P injection reading rules out, for units running
  below their technical minimum (#32)

## v1.10.5 (2026-09-24)

### Fix

- `widen_generation_only_injections` (default on): account for the load box in
  generation-only node injections instead of assuming zero load (#31)

## v1.10.4 (2026-09-24)

### Fix

- `release_load_only_injections` (default on): a load-only node injection on a
  node with declared generation no longer pins the unit output to zero (#30)

## v1.10.3 (2026-09-23)

### Fix

- Treat nodes without active generation as generator-less: nodes whose
  generators are all off lose their generation box and `exist_gen`, reactive-only
  devices (P range exactly `[0, 0]`) get no active-power variable
  (`bounds.is_reactive_only_generation`), and PV nodes without active
  generators are demoted (#29)

## v1.10.2 (2026-09-23)

### Behavior changes

- Transformer series impedance is referred through the actual tap ratio
  (`ybus.series_admittance = |t|²/(r+jx)`). Off-nominal taps used to be modelled
  with an impedance error of `1/|t|²`; lines and nominal-ratio transformers are
  bit-identical. `losses` uses the same referred impedance and the
  `from_pandapower` adapter moves the impedance to the high-voltage side
  accordingly (#28)

## v1.10.1 (2026-09-13)

### Fix

- Raise `exist_gen` on nodes whose generation comes from aggregated active
  generators, so the generation box variable exists (#27)

## v1.10.0 (2026-07-17)

### Feat

- Opt-in `voltage_filter_deactivate_nonphysical` deactivates grossly
  nonphysical voltage readings (`V <= 0`, below `0.5 Vnom` or above `1.5 Vnom`)
  instead of downweighting them
- Opt-in `reconcile_respect_bounds` clips the node-balance reconciliation to
  the node load / generation bounds
- Per-measurement variance scale plans `v_sigma2_scale` and
  `flow_sigma2_scale` in `DerivedInputs`
- Opt-in `ipm_transit_balance_sigma2_pu` (tight zero-injection rows for transit
  nodes; removed again after v1.13.0, see "Unreleased")

All four are off by default and bit-identical to the previous release when
unused (#26)

## v1.9.0 (2026-07-07)

### Feat

- `voltage_filter_min_nominal_kv` (default 110): lowest nominal voltage class
  checked by the voltage range filter; `0` checks every class (#25)

## v1.8.0 (2026-07-07)

### Feat

- Chain-mirror pseudo-V (`v_mirror_chain`, default off) propagates the
  voltage of live measurements into deep chains with a per-hop decay (#22)
- Shunt-sanity research flag (`shunt_sanity`, default off) detects defective
  shunt records by trying off / flip variants in full pipeline reruns (#21)

### Perf

- `refine_merge` (default off): the V refine family shares one warm re-solve
  (wall time -19 to -43 % on the largest models) (#23)

### Behavior changes

- `worst_residuals` defaults to `scope="real"` and ranks telemetry only;
  `scope="all"` restores the previous content, and unranked rows skip the
  triangular substitution (#24)

## v1.7.0 (2026-07-06)

### Refactor

- Consistency cleanup: shared pipeline helpers instead of five copies of the
  solver-kwargs assembly, a shared refine skeleton, a single step walker,
  shared utilities (`id_to_pos_map`, `floored_sigma2`, `sparse_diag`, ...),
  `MeasurementQuality` / `FilterFlag` enums, SHGM / Huber re-weighting in
  `algebra/estimators.py`, a `WLSResult` dataclass, explicit
  `include_quality_summary` / `quality_summary_top_n` parameters, and a dict
  stats contract for steps (#20)

## v1.6.0 (2026-06-14)

### Feat

- `v_mirror_cross_at` (default off): V-mirror fallback for blind clusters
  behind transformers, restricted to transformers with both sides at 110 kV or
  above (#19)

## v1.5.1 (2026-06-14)

### Perf

- Diagonal `R⁻¹` as column scaling in WLS / IPM, vectorized write-back of
  measurement estimates, and a single σ² floor constant (`SIGMA2_FLOOR`)
  across solvers (#18)

## v1.5.0 (2026-06-12)

### Feat

- v-mirror step lifts blind-cluster pseudo-voltage to measured boundary (#16)

## v1.4.0 (2026-06-12)

### Feat

- V-refine step tightening consistent voltage measurements (#14)

## v1.3.0 (2026-06-11)

### Perf

- CHOLMOD KKT solver with symbolic factorization reuse (optional cvxopt) (#12)

## v1.2.0 (2026-06-11)

### Feat

- bad-data re-pass step on solved-SE residuals (default off) (#10)

### Perf

- compute quality summary once post-hoc, Cholesky-based residuals, vectorized jacobian assembly (#8)

## v1.1.0 (2026-06-11)

### Feat

- node-balance reconciliation, real-flow σ-floor, and config↔derived guard (#7)

## v1.0.0 (2026-06-10)

### Feat

- IPM Q-physics, two-level convergence status, NaN measurements, contract 2.0.0 (#5)

## v0.2.0 (2026-06-05)

### Feat

- Canonical-only input contract: the domain tables (`nodes`, `branches`,
  `measurements`, `generators`, `tap_steps`, `load_characteristics`, `shunts`)
  are first-class `.npz` keys and the `raw_tables` tier is removed (breaking
  `.npz` layout); O(n) bulk measurement insertion (#4)

## v0.1.0 (2026-06-03)

### Feat

- First public release: WLS / IPM state estimator on numpy and scipy behind a
  versioned `.npz` data contract; type checking (mypy) is a required CI gate
  and the release pipeline builds and smoke-tests the wheel (#1, #2, #3)
