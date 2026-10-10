"""Public API of the SE module.

A single entry point — ``estimate()``. Dispatch on ``algorithm`` routes the call
to the corresponding implementation in ``gridstate/algorithms/``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gridstate.algebra.base import BaseAlgebra
from gridstate.algebra.estimators import build_branch_pq_huber_mask
from gridstate.algorithms.ipm import IPMResult, solve_ipm
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.algorithms.wls import solve_wls
from gridstate.post_processing import (
    apply_load_characteristic,
    reconcile_node_balance,
    write_measurement_estimates,
    write_node_estimates,
    write_node_estimates_from_inj,
)
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.quality_summary import (
    compute_chi2,
    observability_warnings_from_H,
    top_worst_imbalance,
    top_worst_residuals,
)
from gridstate.result import SEResult, extract_output_tables
from gridstate.state import (
    InitialPoint,
    StateLayout,
    flat_start,
    flat_start_with_box,
    pack,
    unpack,
    unpack_full,
)
from gridstate.units import BASE_MVA, model_to_pu, write_results_to_model
from gridstate.utils import floored_sigma2, id_to_pos_map
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from gridstate.units import NetworkPU
    from gridstate.working import Working, _ArrayCollection
    from gridstate.z_vector import MeasurementIndex


logger = logging.getLogger(__name__)


Algorithm = Literal["wls", "ipm"]
InitMode = Literal["flat", "results", "slack"]
ZeroInjectionMode = Literal["aux_bus", "no_inj_bus", "zero_pwr_bus"]


def estimate(
    model: Working,
    measurements: _ArrayCollection | None = None,
    algorithm: Algorithm = "wls",
    init: InitMode = "flat",
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    zero_injection: ZeroInjectionMode | None = None,
    huber_c: float = 1.5,
    huber_use_mad: bool = False,
    reconcile_balance: bool = True,
    reconcile_respect_bounds: bool = False,
    kkt_solver: str = "auto",
    include_quality_summary: bool = True,
    quality_summary_top_n: int = 10,
    quality_summary_scope: str = "real",
    x0: InitialPoint | None = None,
    **ipm_kwargs: Any,
) -> SEResult:
    """Run state estimation for a model and telemetry.

    ``model`` is updated in place: ``voltage_magnitude`` (kV) / ``voltage_angle`` (rad) are
    written into the ``NodeCollection``; ``power_from_p/q`` / ``power_to_p/q`` (MW/MVAr) and
    ``current_from/to`` (A) into the
    ``BranchCollection``.

    Args:
        model: working network model — the carrier of the contract input tables.
        measurements: measurement collection. If ``None``,
            ``model.measurements``.
        algorithm: SE algorithm: ``"wls"`` (Gauss-Newton) or ``"ipm"``
            (primal log-barrier with box load/generation variables).
        init: initial guess strategy:
            - ``"flat"`` — V=1 p.u., δ=0 for all nodes;
            - ``"results"`` — take the current ``voltage_magnitude``/``voltage_angle``
              from ``model``;
            - ``"slack"`` — V=V_slack for all nodes, δ=0.
        x0: explicit initial point (:class:`gridstate.state.InitialPoint`,
            V in kV / δ in rad keyed by node id). The nodes it covers start
            from it (V converted with their ``voltage_nominal``; a
            non-positive V starts at 1.0 p.u.), the others from ``init``.
            Unlike seeding ``voltage_magnitude``/``voltage_angle`` into the
            node table, ``x0`` changes only the solver's starting point: the
            measurements and pseudo-measurements are untouched. Box variables
            of the IPM start from the node table as usual. ``None`` (default):
            the start is defined by ``init`` alone.
        tolerance: stopping criterion on ``max|ΔE|``.
        max_iterations: maximum number of Gauss-Newton iterations.
        zero_injection: how to handle nodes without injection (not yet
            used — a placeholder for future use).
        reconcile_balance: close the nodal imbalance of the estimates with a final
            post-pass (``gen_est − load_est ≡ p/q_inj_calc``, see
            :func:`gridstate.post_processing.reconcile_node_balance`).
            Default ``True`` — the SE output is consistent as a state (PF input,
            promotion). V/δ and convergence are not affected.
        reconcile_respect_bounds: clip the drain of the reconcile remainder to the
            physical node bounds (``load/gen_*_min/max``; transit → 0).
            Default ``False`` — the former behaviour (unconditional drain).
            See ``reconcile_node_balance(respect_bounds=...)``.
        kkt_solver: solver of the Newton systems (``"auto"`` | ``"cholmod"`` |
            ``"scipy"``, see ``gridstate.algorithms.kkt_solver``). ``auto``
            uses CHOLMOD when cvxopt is installed (×8-11 on large
            models), otherwise scipy spsolve (the former behaviour, bit for bit).
        huber_c: SHGM-IRLS tuning constant; 0 disables robust reweighting
            (see ``gridstate.algebra.estimators``).
        huber_use_mad: normalize SHGM residuals by the MAD scale.
        include_quality_summary: compute chi2/worst_* diagnostics on the
            final solution. Disable in loops/tests where the summary is not
            needed — on large models it costs about as much as a solve.
        quality_summary_top_n: row count of worst_residuals/worst_imbalance.
        quality_summary_scope: ``"real"`` (default) — worst_residuals ranks
            only real (telemetry) measurements; pseudo-priors otherwise crowd
            out telemetry (large models: 8/10 of the top are our own priors
            with r_N in the hundreds) and the masked block-solve is ~2.5×
            cheaper. ``"all"`` — legacy semantics (every z-vector row).
        **ipm_kwargs: IPM-mode knobs, ignored for WLS. ``trust_region_scope``
            (``"all"`` default | ``"state"``) selects the coordinates the IPM
            trust region is measured on (see ``_run_ipm``); everything else is
            forwarded to ``build_ipm_setup`` (``balance_weight_factor``,
            ``bound_relax``, prior sigmas — the A/B-calibration knobs).

    Returns:
        ``SEResult`` with the fields ``success``, ``iterations``, ``objective_value``
        and a reference to the updated ``model``.

    Raises:
        NotImplementedError: for unsupported ``algorithm`` or
            ``zero_injection``.
    """
    if algorithm not in ("wls", "ipm"):
        raise NotImplementedError(
            f"Algorithm {algorithm!r} is not implemented yet; available: 'wls' and 'ipm'."
        )
    if zero_injection is not None:
        raise NotImplementedError(f"zero_injection={zero_injection!r} is not implemented yet.")

    if measurements is None:
        measurements = model.measurements

    # 1. Conversion to p.u.
    network_pu = model_to_pu(model)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)

    # 2. Y-bus, Yf, Yt
    ybus, yf, yt = build_ybus(network_pu)

    # 3. Measurement vector
    z, r_matrix, meas_index = build_z_and_r(model, measurements, network_pu)

    # 4. Initial guess E
    e_init = _build_initial_state(model, network_pu, layout, init, x0=x0)

    # Newton-system solver: one per estimate — reuse of the symbolic
    # factorization between iterations (the structure of G is unchanged).
    kkt = KKTSolver(kkt_solver)

    if algorithm == "ipm":
        # IPM mode: extend layout/state/measurements with box variables and
        # nodal balance equations. ipm_kwargs are passed on to build_ipm_setup
        # (prior_sigma2_*, balance_weight_factor, bound_relax, ...) for A/B
        # calibration runs.
        ipm_res = _run_ipm(
            model,
            network_pu,
            ybus,
            yf,
            yt,
            z,
            r_matrix,
            meas_index,
            layout,
            e_init,
            tolerance=tolerance,
            max_iterations=max_iterations,
            huber_c=huber_c,
            huber_use_mad=huber_use_mad,
            kkt=kkt,
            **ipm_kwargs,
        )
        # _run_ipm has already written *_estimated into model.nodes; the layout
        # changed inside, so for unpack we use the V/δ prefix of the state vector.
        e_final = ipm_res.x
        success = ipm_res.success
        iterations = ipm_res.iterations_outer
        objective = ipm_res.objective_data
        algo_message = ipm_res.message
        convergence_status = ipm_res.status
        delta, v_pu = unpack(e_final[: 2 * network_pu.n_bus - 1], layout)
    else:
        # 5. WLS
        wls_res = solve_wls(
            e_init=e_init,
            z=z,
            r_matrix=r_matrix,
            ybus=ybus,
            yf=yf,
            yt=yt,
            meas_index=meas_index,
            layout=layout,
            network_pu=network_pu,
            tolerance=tolerance,
            max_iterations=max_iterations,
            huber_c=huber_c,
            huber_use_mad=huber_use_mad,
            kkt_solver=kkt,
        )
        e_final = wls_res.x
        success = wls_res.success
        iterations = wls_res.iterations
        objective = wls_res.objective

        # 6. Unpack the state and write it back into the model
        delta, v_pu = unpack(e_final, layout)
        algo_message = ""
        convergence_status = "converged" if success else "not_converged"
    write_results_to_model(model, v_pu, delta, network_pu, yf=yf, yt=yt, ybus=ybus)
    # 7. Post-processing of measurements: estimated_si/value/residual.
    write_measurement_estimates(
        model=model,
        measurements=measurements,
        v_pu=v_pu,
        delta_rad=delta,
        network_pu=network_pu,
        ybus=ybus,
        yf=yf,
        yt=yt,
        meas_index=meas_index,
        z=z,
    )
    # 8. In WLS mode spread p_inj_calc over load_*_estimated /
    # generation_*_estimated. For IPM this is already done via box-vars in
    # _run_ipm (write_node_estimates), no need to repeat.
    if algorithm == "wls":
        write_node_estimates_from_inj(model)
        # If the model contains load characteristics (``load_characteristics``) and a node refers to
        # one, override load_*_estimated with the P(V)/Q(V) polynomial.
        apply_load_characteristic(model)

    # 9. Finalization of the spreading: close the residual nodal imbalance
    # (soft IPM / clips and the load-characteristic override of the WLS spreading) so that the SE output is
    # a consistent state: gen_est − load_est ≡ p/q_inj_calc. Last —
    # after all edits of *_estimated, before extract_output_tables.
    if reconcile_balance:
        reconcile_stats = reconcile_node_balance(model, respect_bounds=reconcile_respect_bounds)
        logger.debug("reconcile_node_balance: %s", reconcile_stats)

    result = SEResult(
        model=model,
        success=success,
        iterations=iterations,
        objective_value=objective,
        algorithm=algorithm,
        v_pu=v_pu,
        delta_rad=delta,
        convergence_status=convergence_status,
        # For IPM the solver diagnostics are passed through (μ_final, |grad|∞ and
        # the stop reason) — previously they were lost in _run_ipm, and the UI saw
        # only a bare "did not converge".
        message=algo_message
        if algo_message
        else ("" if success else f"Did not converge in {iterations}/{max_iterations} iterations"),
        # Output container: all results keyed by id (nodes/branches/measurements),
        # extracted from the model after writing. The canonical Output contract.
        outputs=extract_output_tables(model),
    )

    if include_quality_summary:
        _populate_quality_summary(
            result,
            model=model,
            measurements=measurements,
            network_pu=network_pu,
            ybus=ybus,
            yf=yf,
            yt=yt,
            z=z,
            r_matrix=r_matrix,
            meas_index=meas_index,
            layout=layout,
            v_pu=v_pu,
            delta_rad=delta,
            top_n=quality_summary_top_n,
            scope=quality_summary_scope,
        )

    return result


def _populate_quality_summary(
    result: SEResult,
    *,
    model: Working,
    measurements: _ArrayCollection,
    network_pu: NetworkPU,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    z: np.ndarray,
    r_matrix: csr_matrix,
    meas_index: MeasurementIndex,
    layout: StateLayout,
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    top_n: int,
    scope: str = "real",
) -> None:
    """Fill ``result.chi2 / worst_residuals / worst_imbalance /
    observability_warnings`` after ``write_results_to_model``.

    All errors in the summary are swallowed into the logs: report quality is not an indicator of
    SE success, and a failure here must not break ``estimate()``.
    """
    try:
        if z.shape[0] == 0:
            # No measurements — leave the default empty.
            result.worst_imbalance = top_worst_imbalance(model, n=top_n)
            return

        algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
        h_pu = algebra.evaluate_h(v_pu, delta_rad)
        r_vec = z - h_pu
        H = algebra.evaluate_jacobian(v_pu, delta_rad)

        sigma2 = floored_sigma2(r_matrix.diagonal())

        # Marking pseudo-priors in z order: meas_id → is_pseudo from the collection.
        is_pseudo_z: np.ndarray | None = None
        meas_arr = measurements.to_numpy()
        if meas_arr.size and "is_pseudo" in (meas_arr.dtype.names or ()):
            pseudo_by_id = dict(
                zip(meas_arr["id"].tolist(), meas_arr["is_pseudo"].tolist(), strict=True)
            )
            is_pseudo_z = np.array(
                [bool(pseudo_by_id.get(int(mid), False)) for mid in meas_index.meas_id],
                dtype=bool,
            )

        result.chi2 = compute_chi2(r_vec, sigma2, n_state=int(layout.size))
        result.worst_residuals = top_worst_residuals(
            r_vec,
            sigma2,
            H,
            meas_index,
            z,
            n=top_n,
            network_pu=network_pu,
            is_pseudo=is_pseudo_z,
            scope=scope,
        )
        result.worst_imbalance = top_worst_imbalance(model, n=top_n)
        result.observability_warnings = observability_warnings_from_H(
            H,
            network_pu,
            n_bus=int(network_pu.n_bus),
            non_slack_idx=layout.non_slack_idx,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("quality_summary failed: %s", exc)


def populate_quality_summary(result: SEResult, *, top_n: int = 10, scope: str = "real") -> None:
    """Compute the quality summary post hoc — from the final ``result.model``.

    Does the same as ``estimate(..., include_quality_summary=True)``, but on a
    ready result: rebuilds the p.u. network/Y-bus/z-vector from the current
    state of ``result.model`` and fills ``result.chi2 / worst_residuals /
    worst_imbalance / observability_warnings``. Needed by the pipeline: intermediate
    solves (anti-overshoot, bad-data re-pass) run without a summary, while the summary
    is computed once — on the final solution. The measurements and V/δ
    of the model must correspond to the final solve (in the pipeline they do:
    the revert branches roll back both measurements and V/δ).
    """
    model = result.model
    measurements = model.measurements
    network_pu = model_to_pu(model)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)
    ybus, yf, yt = build_ybus(network_pu)
    z, r_matrix, meas_index = build_z_and_r(model, measurements, network_pu)
    _populate_quality_summary(
        result,
        model=model,
        measurements=measurements,
        network_pu=network_pu,
        ybus=ybus,
        yf=yf,
        yt=yt,
        z=z,
        r_matrix=r_matrix,
        meas_index=meas_index,
        layout=layout,
        v_pu=result.v_pu,
        delta_rad=result.delta_rad,
        top_n=top_n,
        scope=scope,
    )


def _build_initial_state(
    model: Working,
    network_pu: NetworkPU,
    layout: StateLayout,
    init: InitMode,
    x0: InitialPoint | None = None,
) -> np.ndarray:
    """Build ``e_init`` according to the chosen strategy, then overlay ``x0``."""
    e_init = _initial_state_from_mode(model, network_pu, layout, init)
    if x0 is None:
        return e_init
    delta, v_pu = unpack(e_init, layout)
    _overlay_initial_point(model, network_pu, x0, v_pu=v_pu, delta=delta)
    return pack(delta, v_pu, layout)


def _overlay_initial_point(
    model: Working,
    network_pu: NetworkPU,
    x0: InitialPoint,
    *,
    v_pu: np.ndarray,
    delta: np.ndarray,
) -> None:
    """Write the V/δ of ``x0`` into ``v_pu``/``delta`` (bus order) in place.

    Same conversion as ``init="results"`` reading the node table: a node
    covered by ``x0`` starts exactly where it would if its table V/δ had been
    replaced by the ``x0`` values.
    """
    point = x0.as_dict()
    nodes_arr = model.nodes.to_numpy()
    id_to_pos = id_to_pos_map(network_pu.bus_ids)
    for row in nodes_arr[nodes_arr["status"]]:
        got = point.get(int(row["id"]))
        if got is None:
            continue
        pos = id_to_pos[int(row["id"])]
        vm, va = got
        vn = float(row["voltage_nominal"])
        v_pu[pos] = vm / vn if vm > 0 and vn > 0 else 1.0
        delta[pos] = va


def _initial_state_from_mode(
    model: Working,
    network_pu: NetworkPU,
    layout: StateLayout,
    init: InitMode,
) -> np.ndarray:
    """``e_init`` of the ``init`` strategy alone."""
    if init == "flat":
        return flat_start(layout)

    if init == "results":
        nodes_arr = model.nodes.to_numpy()
        active = nodes_arr[nodes_arr["status"]]
        # Renumber to the order of network_pu.bus_ids.
        id_to_pos = id_to_pos_map(network_pu.bus_ids)
        v_pu = np.ones(network_pu.n_bus, dtype=np.float64)
        delta = np.zeros(network_pu.n_bus, dtype=np.float64)
        for row in active:
            pos = id_to_pos[int(row["id"])]
            vm = float(row["voltage_magnitude"])
            vn = float(row["voltage_nominal"])
            if vm > 0 and vn > 0:
                v_pu[pos] = vm / vn
            delta[pos] = float(row["voltage_angle"])
        return pack(delta, v_pu, layout)

    if init == "slack":
        # V_slack is taken from the model (or 1.0 p.u. if not set).
        slack_pos = layout.slack_idx
        slack_id = int(network_pu.bus_ids[slack_pos])
        slack_node = model.nodes.get_by_id(slack_id)
        vm = float(slack_node.voltage_magnitude) if slack_node is not None else 0.0
        vn = float(network_pu.bus_vn_kv[slack_pos])
        v_init = vm / vn if vm > 0 and vn > 0 else 1.0
        v = np.full(network_pu.n_bus, v_init, dtype=np.float64)
        delta = np.zeros(network_pu.n_bus, dtype=np.float64)
        return pack(delta, v, layout)

    raise ValueError(f"Unknown init mode init={init!r}")


def _run_ipm(
    model: Working,
    network_pu: NetworkPU,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    z: np.ndarray,
    r_matrix: csr_matrix,
    meas_index: MeasurementIndex,
    layout_base: StateLayout,
    e_init_base: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    huber_c: float = 0.0,
    huber_use_mad: bool = False,
    huber_skip_transformers: bool = True,
    huber_leverage_b_threshold_pu: float = 2.0,
    huber_w_floor: float = 0.05,
    kkt: KKTSolver | None = None,
    trust_region_scope: str = "all",
    **ipm_kwargs: Any,
) -> IPMResult:
    """IPM mode: extends the state vector with box-vars and solves the primal log-barrier WLS.

    Returns the whole :class:`IPMResult` (``x`` of length ``layout_ipm.size``
    including the box sections, the two-level ``status``, the μ/grad diagnostics).
    Writes the box-var values from the result into ``model.nodes`` via
    ``write_node_estimates``.

    ``trust_region_scope`` — the coordinates whose Newton step is bounded by
    the trust region (``tr_radius=0.5``): ``"all"`` (default) — the whole state
    vector, box variables included; ``"state"`` — V and δ only. In both cases
    the step keeps its direction (one scaling factor for the whole step); with
    ``"state"`` the factor comes from V/δ, so a generation/load box variable
    that has to move tens of p.u. no longer shrinks the V/δ step. The box
    variables stay protected by the fraction-to-boundary rule. Note that the
    Huber reweighting starts after ``huber_warmup_iters`` inner iterations, so
    a solve that converges in fewer steps may finish before it engages.
    """
    if trust_region_scope not in ("all", "state"):
        raise ValueError(
            f"trust_region_scope={trust_region_scope!r} is not one of ['all', 'state']"
        )
    setup = build_ipm_setup(
        model,
        network_pu,
        z,
        r_matrix,
        meas_index,
        layout_base=layout_base,
        **ipm_kwargs,
    )
    layout_ipm = setup.layout

    # Starting state: V/δ from e_init_base + box-vars from the initial values of the node table.
    delta_init, v_init = unpack(e_init_base, layout_base)
    e_init_ipm = flat_start_with_box(
        layout_ipm,
        pgen_init=setup.pgen_init,
        qgen_init=setup.qgen_init,
        pnag_init=setup.pnag_init,
        qnag_init=setup.qnag_init,
    )
    # Overwrite δ, V from e_init_base (if init="results" — it is not flat there).
    e_init_ipm[layout_ipm.offset_delta : layout_ipm.offset_delta + layout_ipm.n_bus - 1] = (
        delta_init[layout_ipm.non_slack_idx]
    )
    e_init_ipm[layout_ipm.offset_v : layout_ipm.offset_v + layout_ipm.n_bus] = v_init

    algebra = BaseAlgebra(ybus, yf, yt, setup.meas_index, layout_ipm, network_pu)

    r_inv_diag = 1.0 / floored_sigma2(setup.r_matrix.diagonal())

    # SHGM-IRLS mask shared with solve_wls; balance/prior rows appended by
    # build_ipm_setup are never reweighted (padding inside the helper).
    huber_mask = None
    if huber_c > 0.0:
        mask = build_branch_pq_huber_mask(
            setup.meas_index,
            network_pu,
            m_total=int(setup.z.shape[0]),
            skip_transformers=huber_skip_transformers,
            leverage_b_threshold_pu=huber_leverage_b_threshold_pu,
        )
        huber_mask = mask if mask.any() else None

    def residual_fn(x: np.ndarray) -> np.ndarray:
        delta, v, pgen, qgen, pnag, qnag = unpack_full(x, layout_ipm)
        h = algebra.evaluate_h(
            v,
            delta,
            pgen_estimated=pgen,
            qgen_estimated=qgen,
            pnag_estimated=pnag,
            qnag_estimated=qnag,
        )
        residual: np.ndarray = setup.z - h
        return residual

    def jacobian_fn(x: np.ndarray) -> csr_matrix:
        delta, v, _pg, _qg, _pn, _qn = unpack_full(x, layout_ipm)
        return algebra.evaluate_jacobian(v, delta)

    result = solve_ipm(
        x_init=e_init_ipm,
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        r_inv_diag=r_inv_diag,
        box_idx=setup.box_idx_in_state,
        box_lo=setup.box_lo,
        box_hi=setup.box_hi,
        inner_tol=max(tolerance, 1e-4),
        inner_max=max_iterations,
        outer_max=20,
        mu_init=0.1,
        tr_radius=0.5,
        # The box sections are the only boxed coordinates of layout_ipm, so
        # "outside the boxes" is exactly V/δ.
        tr_exclude_box=trust_region_scope == "state",
        huber_c=huber_c,
        huber_mask=huber_mask,
        huber_w_floor=huber_w_floor,
        huber_use_mad=huber_use_mad,
        # On average IPM makes 1 inner step per outer (lazy-outer break). Over
        # 9-12 outers on large models warmup=5 (as in WLS) gives Huber
        # 0-3 effective iterations. We reduce it to 2 — Huber activates
        # from the 3rd outer iteration, leaving headroom before μ_min for the downweight of
        # outliers.
        huber_warmup_iters=2,
        # adaptive_k=2.0 (Abur & Exposito ch.6 standard): c_eff =
        # max(c, 2·median(|r/σ|)). In solve_ipm adaptive_applied is reset
        # on every outer — c_eff is recomputed from the current residuals.
        huber_adaptive_k=2.0,
        kkt_solver=kkt,
    )

    # Extract the box-vars and write them into the node table (MW/MVAr = p.u. * BASE_MVA).
    _, _, pgen, qgen, pnag, qnag = unpack_full(result.x, layout_ipm)

    if pgen.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.pgen_node_pos],
            generation_p=pgen * BASE_MVA,
        )
    if qgen.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.qgen_node_pos],
            generation_q=qgen * BASE_MVA,
        )
    if pnag.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.pnag_node_pos],
            load_p=pnag * BASE_MVA,
        )
    if qnag.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.qnag_node_pos],
            load_q=qnag * BASE_MVA,
        )

    return result
