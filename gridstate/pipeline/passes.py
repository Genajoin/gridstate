"""Solver calls of the pipeline: the main solve and the post-solve re-solve passes.

Every call of :func:`gridstate.api.estimate` made by the pipeline goes through
this module: the ``estimate`` step and the warm re-solves of the post-solve
passes (bad-data re-pass, V refine, V mirror, chain mirror, merged refine,
anti-overshoot). The solver keyword arguments are assembled in one place
(:func:`_estimate_kwargs`), and the two-pass passes share one skeleton
(:func:`_refine_two_pass`) built on the acceptance policy of
:mod:`gridstate.resolve_pass`.

Tests that need to intercept the solver or a classifier patch the names of
this module (``gridstate.pipeline.passes.estimate``,
``gridstate.pipeline.passes.classify_v_mirror``, ...): this is where they are
looked up at call time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gridstate.api import estimate
from gridstate.bad_data_repass import apply_bad_data_plan, classify_bad_data
from gridstate.pipeline.config import PipelineConfig
from gridstate.pipeline.context import _Ctx
from gridstate.post_processing import refine_anti_overshoot
from gridstate.resolve_pass import ResolveGuard
from gridstate.result import SEResult
from gridstate.v_mirror import apply_v_mirror_plan, classify_chain_mirror, classify_v_mirror
from gridstate.v_refine import apply_v_refine_plan, classify_v_refine
from gridstate.working import table_view


# ---------------------------------------------------------------------------
# Solver keyword arguments
# ---------------------------------------------------------------------------


def _effective_huber_c(cfg: PipelineConfig) -> float:
    if cfg.huber_c is not None:
        return float(cfg.huber_c)
    return 2.0 if cfg.algorithm == "ipm" else 1.5


def _ipm_kwargs(cfg: PipelineConfig) -> dict:
    return {
        "balance_weight_factor": cfg.ipm_balance_weight_factor,
        "prior_sigma2_bus_equiv_pu": cfg.ipm_prior_sigma2_bus_equiv_pu,
        "boxed_balance_weight_factor": cfg.ipm_boxed_balance_weight_factor,
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
    }
    kw.update(overrides)
    if cfg.algorithm == "ipm":
        kw.update(_ipm_kwargs(cfg))
    return kw


# ---------------------------------------------------------------------------
# Main solve
# ---------------------------------------------------------------------------


def _s_estimate(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    ctx.result = estimate(ctx.model, **_estimate_kwargs(cfg, init=cfg.init))
    return {
        "algorithm": cfg.algorithm,
        "success": bool(ctx.result.success),
        "iterations": int(ctx.result.iterations),
        "objective_value": float(ctx.result.objective_value),
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


def _refine_two_pass(
    ctx: _Ctx,
    *,
    classify: Callable[[_Ctx], Any],
    apply: Callable[[_Ctx, Any], dict],
    empty_stats: Callable[[Any], dict],
    unusable_reason: str,
) -> dict:
    """Shared skeleton of the two-pass post-solve steps.

    Guard on a usable first-pass solution, classify residuals into a plan,
    apply the plan to the working model and warm re-solve. The steps differ
    only in their classifier/applier and skip wording; the plan object must
    expose an ``empty`` property. A re-solve that loses convergence is
    reverted (see :mod:`gridstate.resolve_pass`).
    """
    assert ctx.result is not None  # steps run after estimate → result exists
    if not ctx.result.success:
        # An unusable solution (completed=False) has unreliable residuals —
        # classifying on them is worse than skipping the re-pass.
        return {"skipped": unusable_reason}
    plan = classify(ctx)
    if plan.empty:
        return empty_stats(plan)
    guard = ResolveGuard(ctx.model, ctx.result)
    guard.arm()
    stats = apply(ctx, plan)
    result = _warm_resolve(ctx)
    stats.update(
        {
            "success": bool(result.success),
            "iterations": int(result.iterations),
        }
    )
    # A non-converged re-solve must not replace the converged first pass:
    # restore the measurement edits, V/δ and the previous result.
    ctx.result = guard.settle(result, stats)
    return stats


def _s_bad_data_repass(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_bad_data(
            table_view(c.model.measurements),
            table_view(c.model.branches),
            threshold=cfg.bad_data_threshold,
            sigma_cap=cfg.bad_data_sigma_cap,
            flip_ratio=cfg.bad_data_flip_ratio,
        ),
        apply=lambda c, plan: apply_bad_data_plan(
            c.model, plan, damp_factor=cfg.bad_data_damp_factor
        ),
        empty_stats=lambda plan: {"candidates": plan.n_candidates, "skipped": "no-op (план пуст)"},
        unusable_reason="решение непригодно — нет надёжных остатков",
    )


_MERGED_SKIP = {"skipped": "слит в refine_merged (refine_merge=True)"}


def _s_v_refine(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    if cfg.refine_merge:
        return dict(_MERGED_SKIP)
    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_v_refine(
            table_view(c.model.measurements),
            rn_threshold=cfg.v_refine_rn,
        ),
        apply=lambda c, plan: apply_v_refine_plan(c.model, plan, factor=cfg.v_refine_factor),
        empty_stats=lambda plan: {
            "consistent": plan.n_consistent,
            "skipped": "no-op (нет real V-мер)",
        },
        unusable_reason="решение непригодно — нет надёжных остатков",
    )


def _s_v_mirror(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    if cfg.refine_merge:
        return dict(_MERGED_SKIP)
    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_v_mirror(
            table_view(c.model.measurements),
            table_view(c.model.branches),
            table_view(c.model.nodes),
            max_pu_dev=cfg.v_mirror_max_pu_dev,
            min_lift=cfg.v_mirror_min_lift,
            cross_at=cfg.v_mirror_cross_at,
        ),
        apply=lambda c, plan: apply_v_mirror_plan(c.model, plan),
        empty_stats=lambda _plan: {
            "clusters": 0,
            "skipped": "no-op (нет слепых кластеров с границей)",
        },
        unusable_reason="решение непригодно — нет надёжного уровня границы",
    )


def _s_v_mirror_chain(ctx: _Ctx) -> dict:
    cfg = ctx.cfg
    if cfg.refine_merge:
        return dict(_MERGED_SKIP)

    def _apply(c: _Ctx, plan: Any) -> dict:
        stats = apply_v_mirror_plan(c.model, plan)
        # The n_clusters field of a chain plan carries the maximum front depth.
        return {"max_depth": stats.pop("clusters"), **stats}

    return _refine_two_pass(
        ctx,
        classify=lambda c: classify_chain_mirror(
            table_view(c.model.measurements),
            table_view(c.model.branches),
            table_view(c.model.nodes),
            max_pu_dev=cfg.v_mirror_max_pu_dev,
            min_lift=cfg.v_mirror_min_lift,
            decay=cfg.v_mirror_chain_decay,
            max_depth=cfg.v_mirror_chain_max_depth,
            cross_at=cfg.v_mirror_cross_at,
        ),
        apply=_apply,
        empty_stats=lambda _plan: {
            "nodes": 0,
            "skipped": "no-op (нет провисших флэт-плейсхолдеров в глубине)",
        },
        unusable_reason="решение непригодно — нет надёжного уровня источников",
    )


def _s_refine_merged(ctx: _Ctx) -> dict:
    """Merged v_refine + v_mirror + v_mirror_chain: all plans are classified on
    one solution, then one shared warm re-solve (the refine_merge performance mode).

    A re-solve that loses convergence reverts all three plans at once
    (see :mod:`gridstate.resolve_pass`).
    """
    cfg = ctx.cfg
    if not cfg.refine_merge:
        return {"skipped": "выключено (refine_merge=False)"}
    assert ctx.result is not None
    if not ctx.result.success:
        return {"skipped": "решение непригодно — нет надёжных остатков"}
    stats: dict[str, Any] = {}
    applied = False
    # Snapshot is taken lazily, right before the first plan edits the model
    # (classification itself is read-only).
    guard = ResolveGuard(ctx.model, ctx.result)
    if cfg.v_refine:
        vr_plan = classify_v_refine(
            table_view(ctx.model.measurements), rn_threshold=cfg.v_refine_rn
        )
        if not vr_plan.empty:
            guard.arm()
            stats["v_refine"] = apply_v_refine_plan(ctx.model, vr_plan, factor=cfg.v_refine_factor)
            applied = True
    if cfg.v_mirror:
        vm_plan = classify_v_mirror(
            table_view(ctx.model.measurements),
            table_view(ctx.model.branches),
            table_view(ctx.model.nodes),
            max_pu_dev=cfg.v_mirror_max_pu_dev,
            min_lift=cfg.v_mirror_min_lift,
            cross_at=cfg.v_mirror_cross_at,
        )
        if not vm_plan.empty:
            guard.arm()
            stats["v_mirror"] = apply_v_mirror_plan(ctx.model, vm_plan)
            applied = True
    if cfg.v_mirror_chain:
        # Classify AFTER applying v_mirror: its nodes are no longer flat, so the
        # plans do not overlap, as in the sequential step order.
        ch_plan = classify_chain_mirror(
            table_view(ctx.model.measurements),
            table_view(ctx.model.branches),
            table_view(ctx.model.nodes),
            max_pu_dev=cfg.v_mirror_max_pu_dev,
            min_lift=cfg.v_mirror_min_lift,
            decay=cfg.v_mirror_chain_decay,
            max_depth=cfg.v_mirror_chain_max_depth,
            cross_at=cfg.v_mirror_cross_at,
        )
        if not ch_plan.empty:
            guard.arm()
            st = apply_v_mirror_plan(ctx.model, ch_plan)
            stats["v_mirror_chain"] = {"max_depth": st.pop("clusters"), **st}
            applied = True
    if not applied:
        return {"skipped": "no-op (все планы пусты)"}
    result = _warm_resolve(ctx)
    stats.update(success=bool(result.success), iterations=int(result.iterations))
    ctx.result = guard.settle(result, stats)
    return stats


def _s_anti_overshoot(ctx: _Ctx) -> dict:
    assert ctx.result is not None  # the step runs after estimate → the result exists
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
