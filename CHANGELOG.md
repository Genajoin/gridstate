# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries follow the [Conventional Commits](https://www.conventionalcommits.org/)
types of the merged pull requests; breaking and behavior-changing items are
listed separately under "Breaking changes" and "Behavior changes". A release is
cut by tagging the merged main commit; the version is derived from that tag.

## v2.9.0 (2026-10-18)

### Feat

- `pseudo_voltage_scope` (`add_pseudo_measurements(voltage_scope=...)`): the
  nodes whose pseudo-V prior keeps its σ. `"all"` (default) keeps the current
  behavior. `"generation"` keeps it only at the slack and the nodes with
  generation (`exist_gen` or an active generator); elsewhere the prior variance
  is multiplied by `v_out_of_scope_loose_factor` (default 1e4, σ × 100) and the
  prior stays for observability only. A prior at the nominal V on a load node
  without a V measurement pulls its estimate down, and the solver makes up the
  difference with the load Q. `"none"` loosens every pseudo-V prior, the slack
  and the generation nodes included: V is then held by the network, the
  measurements and the voltage limits (meant for `ipm` / `ipm_pd`). The step
  reports `v_priors_loose` in both modes.

### Fix

- The UI manifest showed the IPM settings (balance modes, balance weights and
  the BUS-equivalent prior) only for `algorithm="ipm"`, although `ipm_pd` uses
  them too. Their `depends` is now `{"algorithm": ["ipm", "ipm_pd"]}`: a list
  value in `depends` means "one of". `ipm_trust_region_scope` acts in the
  primal solver only and keeps `{"algorithm": "ipm"}`.

## v2.8.0 (2026-10-18)

### Behavior changes

- `init="flat"` starts V consistent with the transformer ratios instead of
  1 p.u. everywhere: `log V` from a least squares over the branches
  (`log V_to − log V_from = −log k`, weight `1/|z|`, `log V_slack = 0`). A node
  whose nominal voltage does not match its transformer ratio (or an off-nominal
  tap on a low-impedance branch) no longer starts with a mismatch of hundreds
  of p.u. Applies to every algorithm; without off-nominal ratios the start is
  unchanged. WLS on an ill-conditioned model may converge to a neighbouring
  solution (on the test models p95 |ΔV| against the reference within 0.05 kV).
- `ipm_pd` voltage guard on by default (`ipm_voltage_guard` /
  `voltage_guard`, ignored by other algorithms): every node V is kept within a
  box from its `voltage_min` / `voltage_max` (unset or implausible limits fall
  back to 0.55 / 1.8 p.u., 0.7–1.5 p.u. with neither set) and one step changes
  V by at most 20 %, an angle by at most 0.6 rad and a branch angle difference
  by at most 0.35 rad. Where a node limit is active the estimate sits on it.

### Fix

- `ipm_pd` with the global hard balance no longer collapses a generator node
  into the low-voltage power-flow branch from a flat start (V → 0, the node
  angle undetermined, a formally converged V ≈ 0.2 p.u.).

### Feat

- `solve_ipm_pd`: `state_box_*` (barrier boxes on state coordinates that stay
  under the trust region and the stopping test, without box damping) and the
  `step_rel_*` / `step_abs_*` / `step_diff_*` step limits.
- `gridstate.state.tap_consistent_voltages`.

## v2.7.0 (2026-10-18)

### Behavior changes

- `ipm_pd` stops the Huber reweighting once the complementarity μ is at or
  below `huber_freeze_mu` (new solver argument, default 1e-4) and keeps the
  last weights. Estimates move slightly (≤ 0.005 kV median |ΔV| on the test
  models, individual nodes up to ~2e-3 p.u.); iteration counts drop.

### Perf

- `ipm_pd` end game with Huber: rows whose residual sits at the threshold
  flipped their weight every iteration, the merit search halved every other
  step and the solve zigzagged at a linear rate. Equality rows (the hard
  transit balance) made it markedly worse: on the test models the hard mode
  needed up to 109 iterations per solve where the soft one needs 18. With the
  weights frozen: 28; the hard mode now runs within ~10–15 % of the soft one.

## v2.6.0 (2026-10-18)

### Behavior changes

- `ipm_pd` gets the 300-iteration budget with equality rows too (it was 100
  before the primal fallback), including the global `ipm_balance_mode="hard"`:
  with the adaptive box damping and the measurement-space stopping test of
  v2.4.1 the long budget converges where the short one fell back.

### Feat

- `ipm_transit_balance_mode` (new, default `soft`): `hard` makes the P and Q
  balance rows of transit nodes (no `exist_gen` / `exist_load`) equality
  constraints of the solver, to `ipm_transit_balance_tol_p_mw` (new, default
  0.01) / `ipm_transit_balance_tol_q_mvar` (new, default 0.1); the other
  balance rows stay soft. The global hard balance overrides it. `IPMSetup`
  gains `eq_rows` / `eq_tol`.

## v2.5.0 (2026-10-18)

### Behavior changes

- After the run, transit nodes (no `exist_gen` / `exist_load`) whose solved
  injection exceeds `transit_injection_warn_mva` (new, default 5 MVA; 0 turns
  it off) are listed in a WARNING: a data conflict, or a generation, load or
  shunt the model does not describe. Estimates are not affected.

### Feat

- `ipm_transit_balance_weight_factor` (new, default 0 — off): the balance row
  of a transit node gets the variance median(σ²_data) / factor (never looser
  than the soft balance). Their injection is known to be zero; with the
  default soft weight a data conflict spreads over them as a fictitious
  injection.

## v2.4.1 (2026-10-18)

### Behavior changes

- `ipm_pd`: the proximal term on the box-variable diagonal adapts to the step
  length: ×10 after a step αp < 0.1, up to `box_prox_max` (new, default 1e-2 of
  the median weight), ÷10 after αp ≥ 0.5, down to `box_prox`. It damps the step
  only; the fixed point is unchanged. Estimates on the test models move by
  ≤ 1.6e-8 p.u. V with unchanged iteration counts.
- `ipm_pd`: a second stopping test, `fit_tol` (new, default 1e-2): the step moves
  no data row by more than that fraction of its sigma (the μ, feasibility and
  equality tests still apply).

### Fix

- `ipm_pd` no longer runs into its iteration limit and the primal fallback when
  box variables are seen only through near-zero-weight rows (loads with
  effectively free injections): such directions have barrier curvature only,
  their steps dwarf the box and the boundary rule cut every step to a tiny αp.
  A state the data do not determine no longer keeps the V/δ step test from
  passing (it could move, or cycle, without changing any estimate).

## v2.4.0 (2026-10-17)

### Behavior changes

- Breaker branches marked by the input format (R = 0, X = 1 Ohm) get
  X = 110·min_x·Vn Ohm with min_x = 4e-6 (≈ 0.044/Vn p.u.) instead of a fixed
  1e-3 p.u.; `breaker_x_pu` (new) sets a fixed p.u. value instead. Estimates
  move on models carrying the marker (≤ 4.5e-3 p.u. V with `ipm_pd` on the
  test models) (#115)
- A solved state with V/Vnom below `implausible_v_ratio` (new, default 0.3)
  at any node is marked not converged whatever the solver reports, with a
  WARNING listing the nodes and hints; a non-converged WLS solve logs its
  largest normalized residuals (#114, #115)

### Feat

- Piecewise static load characteristics: optional input table
  `load_characteristic_segments` (contract 3.2.0) with per-segment
  coefficients up to V⁴ for P and Q; `apply_load_characteristic` uses it in
  place of the quadratic row; without the table nothing changes (#113, #115)

## v2.3.0 (2026-10-17)

### Behavior changes

- `ipm_pd` uses a fixed Huber threshold `c_eff = huber_c` by default
  (`ipm_huber_threshold="fixed"`). The previous per-iteration adaptive raise
  made the estimate depend on the solver path; `"adaptive"` restores it. On
  regional models the quality is unchanged; `ipm_pd` estimates move slightly
  (≤ 6e-5 p.u. V on the test models). `wls` / `ipm` are unaffected (#111)
- `ipm_pd` selects its step by Armijo backtracking on a merit function
  (barrier objective, augmented Lagrangian of the equality rows, slack
  penalty). It now converges on weakly determined cases (e.g. unmeasured
  units with a wide box under `injection_roles`) that needed the primal
  fallback; `ipm_pd` estimates can move slightly. The pipeline's `ipm_pd`
  iteration limit is 300 (100 under a hard balance); the solve stops after
  `max_search_failures` iterations without merit decrease (#106)

### Feat

- Convergence hints: a solve that does not converge, falls back from
  `ipm_pd`, or takes more than `ipm_hint_iterations` iterations with a
  dominant step limiter logs a WARNING naming the limiting nodes / variables
  (with a suggestion), loose unmeasured box variables, the largest normalized
  residuals and part measurements outside their range. `IPMResult` gains
  `step_limiters` / `limiting_rows` (#110)

## v2.2.0 (2026-10-17)

All new behaviour is opt-in; with the defaults the results are unchanged.

### Feat

- `algorithm="ipm_pd"`: Mehrotra predictor-corrector interior-point solver;
  predictor and corrector share one factorization per iteration
  (`KKTSolver.factor_normal`). About 5× less solver time than `ipm` on large
  models, with closely matching estimates. A solve that does not reach its
  KKT test is redone by the primal IPM from the same start (#103)
- `balance_mode="hard"` (`ipm_balance_mode`): node balance rows become
  equality constraints in both IPM solvers (method of multipliers,
  quasi-definite step) with P / Q tolerances `balance_tol_p_mw` /
  `balance_tol_q_mvar`. Experimental: it does not converge on every real
  model yet (#103)
- `fixed_injection_from_input` (`ipm_fixed_injection_from_input`): the input
  injection of node parts without a box variable enters the balance as a
  fixed term. Experimental (#103)
- `measurement_bounds` (`ipm_measurement_bounds`, `ipm_pd` only):
  `min_value` / `max_value` bound the estimated value of any measurement as
  an inequality (#103)
- `injection_roles` (`ipm_injection_roles`, `ipm` / `ipm_pd`): node injection
  rows tagged `source_code` `"inj_gen"` / `"inj_load"`
  (`SOURCE_INJ_GEN` / `SOURCE_INJ_LOAD`) measure the generation / load box
  variable directly; the node box is narrowed by the measurement box, the
  unmeasured part of a measured node gets a prior, and the net row stays only
  where a measured part has no box variable. The telemetry kernel emits the
  part rows when the mode is on; `release_load_only_injections` and
  `widen_generation_only_injections` are skipped in it (#103)
- `DerivedInputs.telemetry_bounds` (derived table `telemetry_bounds`):
  optional plausible ranges of telemetry values, written by the telemetry
  kernel into `min_value` / `max_value` (summed over components for
  injection rows, generation minus load for the net row). Data contract
  3.1.0, additive (#103)
- `ipm_balance_weight_factor` accepts values up to 1e6 (was 10) (#103)

## v2.1.0 (2026-10-14)

### Feat

- `SEResult.normalized_residuals` (`MeasurementNormalizedResiduals`, exported
  from the package root): `r_N` and `Ω_ii / σ²_i` of every z-vector
  measurement by measurement id, in z-vector order. The quality summary keeps
  the vector it already computes; `worst_residuals` and
  `near_critical_measurements` still rank and trim it. Rows outside the
  residual scope (pseudo-measurements under `"real"`) hold `nan`, critical
  measurements `inf`; `None` when the summary is not computed. Additive, no
  data contract change (#101)

## v2.0.0 (2026-10-10)

### Breaking changes

- Data contract 3.0.0 (`CONTRACT_VERSION`, MAJOR bump). It covers the typed
  derived inputs below and the changed meaning of the transformer
  `conductance_from` / `susceptance_from` fields (#52): the version policy now
  states that changing the meaning of an existing field is a MAJOR change, so
  data written under contract 2.x fails `is_data_compatible` instead of being
  read with the new meaning (#65)
- The `.npz` contract stores `DerivedInputs` as typed tables plus JSON
  metadata (`derived.<table>` / `derived.meta`) instead of a pickled blob, and
  the loader never unpickles by default (#65):
  - new public module `gridstate.contract.derived_tables`: the key vocabulary,
    the dtype / shape / meaning of every derived table
    (`DERIVED_TABLES`), and `derived_to_tables` / `derived_from_tables`;
    the writer rejects keys outside the vocabulary with `ValueError`;
  - neutral key vocabulary of the plans: telemetry kinds `v`, `p_from` /
    `p_to` / `q_from` / `q_to`, `p_gen` / `p_load` / `q_gen` / `q_load`, with
    an optional component number for injection components (`"p_gen:3"`,
    formerly a per-unit suffix); topology object kinds `node` / `branch` /
    `generator`; skip reasons `no_value` / `partial_inputs` /
    `evaluation_error`; `materialize_obs` must be `{quantity: {node_id:
    value}}` with `load_p` / `load_q` / `gen_p` / `gen_q`. The step statistics
    keep their names. The kind names are defined once, as `KIND_*` constants
    of `gridstate.contract.derived_tables`; the tables of
    `gridstate.telemetry.kinds` (`KIND_MAP`, `NODE_INJ_MAP`) are keyed by them;
  - `DerivedInputs` fields are typed (`TopologyItem`, `TelemetryKey`,
    `TelemetryValue`, `FlowKey` in `gridstate.contract.derived`);
  - `load_se_input_npz(path, trusted=False)`: a contract 2.x file (pickled
    derived inputs, former transformer shunt meaning) raises
    `UntrustedLegacyFormatError` (a `ValueError`); with `trusted=True` it is
    unpickled, mapped onto the new vocabulary (topology items of other object
    kinds, never applied by the core, are dropped), returned with the current
    contract version and a `LegacyContractWarning` about the transformer
    shunt semantics. Re-save it with `save_se_input` to convert it;
  - the IEEE fixtures are converted to the new format (tables unchanged)
- Transformer branch shunt model: the whole "from" shunt of a branch
  (`conductance_from` / `susceptance_from` plus half of `conductance` /
  `susceptance`) is connected at the "from" node, in the "from"-side base,
  before the ideal transformer. `build_ybus` no longer divides it by `|t|²`
  (`Yff = Ysf + Yc_from`; `Yft`, `Ytf`, `Ytt` are unchanged), so it no longer
  depends on the tap. The meaning of the contract fields `conductance_from` /
  `susceptance_from` on transformers with an off-nominal ratio changes
  accordingly: inputs that relied on the division (a shunt behind the ideal
  transformer) must divide the "from" shunt by `|t|²` themselves. Lines and
  transformers with `|t| = 1` in p.u. are not affected. Branch flows, `h(x)`,
  the Jacobian, branch results and `compute_system_losses` all follow the
  Y-bus. `tap_steps.shunt_factor` is still applied as given, but a tap change
  needs no shunt rescaling any more (use `1.0`). `from_pandapower` converts
  pandapower's from-end shunt (behind the ideal transformer) by dividing it by
  `tap_pu²`; the Y-bus still equals pandapower's to 1e-13 p.u. for both
  `trafo_model="t"` and `"pi"`. The IEEE fixtures and golden snapshots do not
  change: none of their branches combines an off-nominal ratio with a "from"
  shunt (#52)
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
- `gridstate.pipeline` is now a package. It exports `STEPS`, `DerivedInputs`,
  `PipelineConfig`, `Step`, `check_step_order`, `manifest`, `prepare_network`
  and `run`; the private helpers of the former module live in its submodules
  (#98, see the next entry). Patch `gridstate.pipeline.passes` instead of
  `gridstate.pipeline` in tests (#50)
- Invariants that were `assert` statements now raise `ValueError` (missing
  derived plan, naming it) or `RuntimeError` (no solution), also under
  `python -O` (#50)
- Removed the compatibility aliases and re-exports kept during the refactoring
  above, and one deprecated parameter (old → new):
  - module `gridstate.contract.runtime` → `gridstate.runtime` (`run`,
    `prepare_network`; also `gridstate.contract.run` /
    `gridstate.contract.prepare_network`, which stay),
    `gridstate.contract.containers` (`SEInput`, `SEOutput`, `load_se_input`)
    and `gridstate.pipeline.orchestration.shunt_sanity_rerun` (was
    `_shunt_sanity_rerun`; its `pristine=` argument is `model=`);
  - `gridstate.resolve_pass.ResolveGuard` → `ResolvePass`;
  - `gridstate.telemetry.quality.QUALITY_GOOD` / `QUALITY_QUESTIONABLE` /
    `QUALITY_BAD` (also re-exported from `gridstate.telemetry`) and
    `gridstate.validation.bad_data.QUALITY_BAD` →
    `gridstate.constants.MeasurementQuality.GOOD` / `.QUESTIONABLE` / `.BAD`
    (an `IntEnum` with the same values; the classifiers still return plain
    ints);
  - `gridstate.telemetry.units.variance_voltage` / `variance_power` /
    `variance_branch_q` → `gridstate.telemetry.noise` (or the
    `gridstate.telemetry` package); `gridstate.telemetry.units.normalize_guid`
    stays;
  - private module `gridstate.telemetry._specs` (`_KIND_MAP`,
    `_NODE_INJ_MAP`, `_INJ_MT`) → the public, documented
    `gridstate.telemetry.kinds` (`KIND_MAP`, `NODE_INJ_MAP`, `INJ_MT`, now
    read-only mappings);
  - the private helpers re-exported from `gridstate.pipeline` → their
    submodules: `_param`, `_toggle` and the `_G_*` group labels →
    `gridstate.pipeline.config`; `_Ctx` → `.context`; `_type_name` →
    `.manifest`; the pre-solve step adapters `_s_*` → `.preparation`; the
    solve and re-solve step adapters (`_s_estimate`, `_s_bad_data_repass`,
    `_s_v_refine`, `_s_v_mirror`, `_s_v_mirror_chain`, `_s_refine_merged`,
    `_s_anti_overshoot`) and `_MERGED_SKIP`, `_effective_huber_c`,
    `_estimate_kwargs`, `_ipm_kwargs`, `_refine_two_pass`, `_warm_resolve` →
    `.passes`; `_REQUIRED_DERIVED_PLANS`, `_build_working`,
    `_check_derived_consistency`, `_emit`, `_execute_step`,
    `_seed_warm_start` → `.runner`;
  - the deprecated `bound_relax` parameter of `build_ipm_setup` (and of
    `estimate(..., bound_relax=...)` in IPM mode, which now fails with
    `TypeError`) is gone; it had no pipeline setting and its default `0` is
    the only behaviour left (#98).

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
  `pipeline/orchestration.py`; containers in `contract/containers.py`. A single
  accepted shunt candidate reuses its trial result (3 runs instead of 4) (#70)
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
