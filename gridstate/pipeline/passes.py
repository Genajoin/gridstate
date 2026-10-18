"""Solver calls of the pipeline: the main solve and the post-solve re-solve passes.

Every call of :func:`gridstate.api.estimate` made by the pipeline goes through
this module: the ``estimate`` step and the warm re-solves of the post-solve
passes (bad-data re-pass, V refine, V mirror, chain mirror, merged refine,
anti-overshoot). The solver keyword arguments are assembled in one place
(:func:`_estimate_kwargs`). The refine family (bad-data re-pass, V refine,
V mirror, chain mirror) is declared once as :class:`RefineStage` objects;
the sequential steps and the merged step run them through one skeleton
(:func:`_resolve_pass`) built on :class:`gridstate.resolve_pass.ResolvePass`
(guard → snapshot → apply → re-solve → accept | revert), which also backs
the anti-overshoot pass.

Tests that need to intercept the solver or a classifier patch the names of
this module (``gridstate.pipeline.passes.estimate``,
``gridstate.pipeline.passes.classify_v_mirror``, ...): this is where they are
looked up at call time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gridstate.api import estimate
from gridstate.bad_data_repass import apply_bad_data_plan, classify_bad_data
from gridstate.pipeline.config import PipelineConfig
from gridstate.pipeline.context import _Ctx, _no_solution
from gridstate.post_processing import refine_anti_overshoot
from gridstate.resolve_pass import ResolvePass, fail_out_of_range, state_range_violation
from gridstate.result import SEResult
from gridstate.v_mirror import apply_v_mirror_plan, classify_chain_mirror, classify_v_mirror
from gridstate.v_refine import apply_v_refine_plan, classify_v_refine
from gridstate.working import Working, table_view


# ---------------------------------------------------------------------------
# Solver keyword arguments
# ---------------------------------------------------------------------------


def _effective_huber_c(cfg: PipelineConfig) -> float:
    if cfg.huber_c is not None:
        return float(cfg.huber_c)
    return 2.0 if cfg.algorithm in ("ipm", "ipm_pd") else 1.5


def _ipm_kwargs(cfg: PipelineConfig) -> dict:
    return {
        "balance_weight_factor": cfg.ipm_balance_weight_factor,
        "prior_sigma2_bus_equiv_pu": cfg.ipm_prior_sigma2_bus_equiv_pu,
        "boxed_balance_weight_factor": cfg.ipm_boxed_balance_weight_factor,
        "transit_balance_weight_factor": cfg.ipm_transit_balance_weight_factor,
        "transit_balance_mode": cfg.ipm_transit_balance_mode,
        "transit_balance_tol_p_mw": cfg.ipm_transit_balance_tol_p_mw,
        "transit_balance_tol_q_mvar": cfg.ipm_transit_balance_tol_q_mvar,
        "trust_region_scope": cfg.ipm_trust_region_scope,
        "balance_mode": cfg.ipm_balance_mode,
        "balance_tol_p_mw": cfg.ipm_balance_tol_p_mw,
        "balance_tol_q_mvar": cfg.ipm_balance_tol_q_mvar,
        "fixed_injection_from_input": cfg.ipm_fixed_injection_from_input,
        "measurement_bounds": cfg.ipm_measurement_bounds,
        "voltage_guard": cfg.ipm_voltage_guard,
        "injection_roles": cfg.ipm_injection_roles,
        "hint_iterations": cfg.ipm_hint_iterations,
        "huber_threshold": cfg.ipm_huber_threshold,
    }


def _estimate_kwargs(cfg: PipelineConfig, *, init: str, **overrides: Any) -> dict[str, Any]:
    """Solver kwargs shared by every estimate/re-solve call in the pipeline.

    Single assembly point for the solver call: the first pass and every warm
    re-solve (bad-data, v-refine, v-mirror, anti-overshoot) differ only in
    ``init`` and the ``overrides`` they pass (anti-overshoot uses a larger
    iteration budget and its own tolerance).
    """
    kw: dict[str, Any] = {
        "algorithm": cfg.algorithm,
        "init": init,
        "tolerance": cfg.tolerance,
        "max_iterations": cfg.max_iterations,
        "huber_c": _effective_huber_c(cfg),
        "kkt_solver": cfg.kkt_solver,
        # Quality summary is computed once, on the final solution in run()
        # (see populate_quality_summary); intermediate solves skip it.
        "include_quality_summary": False,
        "reconcile_balance": cfg.reconcile_balance,
        "reconcile_respect_bounds": cfg.reconcile_respect_bounds,
        "success_policy": cfg.success_policy,
        "implausible_v_ratio": cfg.implausible_v_ratio,
    }
    kw.update(overrides)
    if cfg.algorithm in ("ipm", "ipm_pd"):
        kw.update(_ipm_kwargs(cfg))
    return kw


# ---------------------------------------------------------------------------
# Main solve
# ---------------------------------------------------------------------------


def _s_estimate(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    kwargs = _estimate_kwargs(cfg, init=cfg.init)
    if ctx.x0 is not None:
        # Warm start in x0_only mode: the previous solution is the start of the
        # main solve only. The re-solve passes start from the table, which by
        # then holds this solve's V/δ, as on a cold start.
        kwargs["x0"] = ctx.x0
    ctx.result = estimate(ctx.model, **kwargs)
    # A state out of range (e.g. angles beyond ±2π) is unusable whatever the
    # solver reports: it cannot be converted back to p.u., so the re-solve
    # passes and the quality summary must not start from it.
    violation = state_range_violation(ctx.model.nodes.to_numpy())
    if violation is not None:
        fail_out_of_range(ctx.result, violation)
    stats = {
        "algorithm": cfg.algorithm,
        "success": bool(ctx.result.success),
        "iterations": int(ctx.result.iterations),
        "objective_value": float(ctx.result.objective_value),
        **_status_stats(ctx.result),
    }
    if violation is not None:
        stats["state_violation"] = violation
    return stats


def _status_stats(result: SEResult) -> dict[str, Any]:
    """Solver-independent status and objective of a solve, for the step stats."""
    return {
        "solve_status": None if result.solve_status is None else result.solve_status.value,
        "objective_wls": float(result.objective_wls),
    }


# ---------------------------------------------------------------------------
# Warm re-solve passes
# ---------------------------------------------------------------------------


def _warm_resolve(ctx: _Ctx, **overrides: Any) -> SEResult:
    """Warm re-solve on the (edited) working model and store it in ``ctx``.

    V/δ of the previous pass are already in the working layer, hence
    ``init="results"``.
    """
    ctx.result = estimate(ctx.model, **_estimate_kwargs(ctx.cfg, init="results", **overrides))
    return ctx.result


@dataclass(frozen=True)
class RefineStage:
    """One classify → apply unit of a two-pass refine step.

    The same stage objects drive both refine paths: the sequential steps (one
    stage, one re-solve each) and the merged step (several stages, one shared
    re-solve). The classifier and the applier read their knobs from the
    config, so the keyword plumbing exists once.

    Attributes:
        name: step name; also the config toggle of the stage and the key of
            its stats in the merged step.
        classify: ``(cfg, model) -> plan`` (read-only; the plan exposes
            ``empty``).
        apply: ``(cfg, model, plan) -> stats`` (edits the working model).
        empty_stats: ``plan -> stats`` of the sequential step when the plan
            is empty.
        unusable_reason: skip reason of the sequential step when the base
            solution did not converge.
    """

    name: str
    classify: Callable[[PipelineConfig, Working], Any]
    apply: Callable[[PipelineConfig, Working, Any], dict]
    empty_stats: Callable[[Any], dict]
    unusable_reason: str


# The lambdas look the library functions up in this module at call time (tests
# patch them here).
_NO_RELIABLE_RESIDUALS = "решение непригодно — нет надёжных остатков"

BAD_DATA_STAGE = RefineStage(
    name="bad_data",
    classify=lambda cfg, model: classify_bad_data(
        table_view(model.measurements),
        table_view(model.branches),
        threshold=cfg.bad_data_threshold,
        sigma_cap=cfg.bad_data_sigma_cap,
        flip_ratio=cfg.bad_data_flip_ratio,
    ),
    apply=lambda cfg, model, plan: apply_bad_data_plan(
        model, plan, damp_factor=cfg.bad_data_damp_factor
    ),
    empty_stats=lambda plan: {"candidates": plan.n_candidates, "skipped": "no-op (план пуст)"},
    unusable_reason=_NO_RELIABLE_RESIDUALS,
)
V_REFINE_STAGE = RefineStage(
    name="v_refine",
    classify=lambda cfg, model: classify_v_refine(
        table_view(model.measurements), rn_threshold=cfg.v_refine_rn
    ),
    apply=lambda cfg, model, plan: apply_v_refine_plan(model, plan, factor=cfg.v_refine_factor),
    empty_stats=lambda plan: {"consistent": plan.n_consistent, "skipped": "no-op (нет real V-мер)"},
    unusable_reason=_NO_RELIABLE_RESIDUALS,
)
V_MIRROR_STAGE = RefineStage(
    name="v_mirror",
    classify=lambda cfg, model: classify_v_mirror(
        table_view(model.measurements),
        table_view(model.branches),
        table_view(model.nodes),
        max_pu_dev=cfg.v_mirror_max_pu_dev,
        min_lift=cfg.v_mirror_min_lift,
        cross_at=cfg.v_mirror_cross_at,
    ),
    apply=lambda _cfg, model, plan: apply_v_mirror_plan(model, plan),
    empty_stats=lambda _plan: {"clusters": 0, "skipped": "no-op (нет слепых кластеров с границей)"},
    unusable_reason="решение непригодно — нет надёжного уровня границы",
)


def _apply_chain(_cfg: PipelineConfig, model: Working, plan: Any) -> dict:
    stats = apply_v_mirror_plan(model, plan)
    # The n_clusters field of a chain plan carries the maximum front depth.
    return {"max_depth": stats.pop("clusters"), **stats}


V_MIRROR_CHAIN_STAGE = RefineStage(
    name="v_mirror_chain",
    classify=lambda cfg, model: classify_chain_mirror(
        table_view(model.measurements),
        table_view(model.branches),
        table_view(model.nodes),
        max_pu_dev=cfg.v_mirror_max_pu_dev,
        min_lift=cfg.v_mirror_min_lift,
        decay=cfg.v_mirror_chain_decay,
        max_depth=cfg.v_mirror_chain_max_depth,
        cross_at=cfg.v_mirror_cross_at,
    ),
    apply=_apply_chain,
    empty_stats=lambda _plan: {
        "nodes": 0,
        "skipped": "no-op (нет провисших флэт-плейсхолдеров в глубине)",
    },
    unusable_reason="решение непригодно — нет надёжного уровня источников",
)


#: Stages of the merged step, in the order of the sequential steps. The chain
#: stage classifies AFTER v_mirror is applied: its nodes are no longer flat,
#: so the plans do not overlap, as in the sequential step order.
MERGED_STAGES: tuple[RefineStage, ...] = (V_REFINE_STAGE, V_MIRROR_STAGE, V_MIRROR_CHAIN_STAGE)


def _resolve_pass(
    ctx: _Ctx,
    stages: Sequence[RefineStage],
    *,
    step: str,
    unusable_reason: str,
    empty_stats: Callable[[Any], dict],
    nest: bool,
) -> dict:
    """Run ``stages`` as one re-solve pass (see :class:`ResolvePass`).

    Guard on a usable solution, classify and apply each stage in order (an
    empty plan is passed over), warm re-solve once, accept or revert. With
    ``nest`` the stats of each applied stage go under its name (merged step);
    otherwise the stats of the (single) stage are the step stats. When no
    plan was applied the step reports ``empty_stats(last plan)`` and the
    model is untouched.
    """
    if ctx.result is None:  # the passes run after estimate
        raise _no_solution(step)
    rp = ResolvePass(ctx.model, ctx.result)
    if not rp.usable:
        # An unusable solution (completed=False) has unreliable residuals —
        # classifying on them is worse than skipping the re-pass.
        return {"skipped": unusable_reason}
    cfg = ctx.cfg
    stats: dict[str, Any] = {}
    plan: Any = None
    applied = False
    for stage in stages:
        plan = stage.classify(cfg, ctx.model)
        if plan.empty:
            continue
        rp.arm()  # snapshot right before the first edit
        stage_stats = stage.apply(cfg, ctx.model, plan)
        if nest:
            stats[stage.name] = stage_stats
        else:
            stats = stage_stats
        applied = True
    if not applied:
        return empty_stats(plan)
    result = _warm_resolve(ctx)
    stats.update(success=bool(result.success), iterations=int(result.iterations))
    stats.update(_status_stats(result))
    # A non-converged re-solve must not replace the converged first pass:
    # restore the measurement edits, V/δ and the previous result.
    ctx.result = rp.settle(result, stats)
    return stats


def _refine_two_pass(ctx: _Ctx, stage: RefineStage) -> dict:
    """Sequential refine step: one stage, one warm re-solve."""
    return _resolve_pass(
        ctx,
        (stage,),
        step="two-pass refine",
        unusable_reason=stage.unusable_reason,
        empty_stats=stage.empty_stats,
        nest=False,
    )


def _s_bad_data_repass(ctx: _Ctx) -> dict:
    return _refine_two_pass(ctx, BAD_DATA_STAGE)


_MERGED_SKIP = {"skipped": "слит в refine_merged (refine_merge=True)"}


def _s_v_refine(ctx: _Ctx) -> dict:
    if ctx.cfg.refine_merge:
        return dict(_MERGED_SKIP)
    return _refine_two_pass(ctx, V_REFINE_STAGE)


def _s_v_mirror(ctx: _Ctx) -> dict:
    if ctx.cfg.refine_merge:
        return dict(_MERGED_SKIP)
    return _refine_two_pass(ctx, V_MIRROR_STAGE)


def _s_v_mirror_chain(ctx: _Ctx) -> dict:
    if ctx.cfg.refine_merge:
        return dict(_MERGED_SKIP)
    return _refine_two_pass(ctx, V_MIRROR_CHAIN_STAGE)


def _s_refine_merged(ctx: _Ctx) -> dict:
    """Merged v_refine + v_mirror + v_mirror_chain: all plans are classified on
    one solution, then one shared warm re-solve (the refine_merge performance mode).

    The enabled stages run in :data:`MERGED_STAGES` order; a re-solve that
    loses convergence reverts all of them at once (see
    :mod:`gridstate.resolve_pass`).
    """
    cfg = ctx.cfg
    if not cfg.refine_merge:
        return {"skipped": "выключено (refine_merge=False)"}
    return _resolve_pass(
        ctx,
        [stage for stage in MERGED_STAGES if getattr(cfg, stage.name)],
        step="refine_merged",
        unusable_reason=_NO_RELIABLE_RESIDUALS,
        empty_stats=lambda _plan: {"skipped": "no-op (все планы пусты)"},
        nest=True,
    )


_STATE_OUT_OF_RANGE = "state out of range (see estimate stats)"


def _s_anti_overshoot(ctx: _Ctx) -> dict:
    """Anti-overshoot pass; its :class:`ResolvePass` lives in
    :func:`gridstate.post_processing.refine_anti_overshoot` (several
    apply/re-solve rounds and its own acceptance criterion)."""
    if ctx.result is None:  # the step runs after estimate
        raise _no_solution("anti_overshoot")
    if state_range_violation(ctx.model.nodes.to_numpy()) is not None:
        # Only the main estimate can leave such a state (the passes revert
        # theirs); no re-solve can start from it.
        return {"skipped": _STATE_OUT_OF_RANGE}
    cfg = ctx.cfg

    def _resolve() -> SEResult:
        # Warm re-solve with extra iteration headroom (80 truncates → phantom
        # non-convergence). WLS tol=1e-4, IPM tol=1e-3: the tolerances the
        # refinement was validated with.
        return _warm_resolve(
            ctx,
            max_iterations=150,
            tolerance=1e-4 if cfg.algorithm == "wls" else 1e-3,
        )

    ctx.result, stats = refine_anti_overshoot(
        ctx.model,
        ctx.result,
        _resolve,
        ceiling=cfg.anti_overshoot_ceiling,
    )
    return dict(stats or {})
