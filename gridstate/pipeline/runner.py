"""Pipeline execution: :func:`run`, :func:`prepare_network` and the event protocol.

Progress is reported through the ``on_event(event: dict)`` callback:
``clone`` / ``warm_start`` / ``step_start`` / ``step_done`` / ``step_skipped`` /
``step_error`` / ``final``. gridstate does not depend on a web framework; the
consumer wraps the events into its own transport (NDJSON and the like).

Event schema
------------
This is the protocol as emitted; it is descriptive, not a promise of a
cleaner one. Every event is a ``dict`` with a ``"type"`` key:

``warm_start``
    ``{"type", "seeded_nodes": int}``. :func:`run` with ``init_state`` only;
    emitted first, before ``clone``.
``clone``
    ``{"type", "nodes": int, "branches": int}``: sizes of the working copy.
    :func:`run` only (not :func:`prepare_network`).
``step_skipped``
    ``{"type", "name", "reason": str}`` for a step of :data:`STEPS` whose
    toggle is off (``reason`` = ``"отключено (<toggle>=False)"``) or that
    needs derived inputs that were not given (``"no derived inputs"``). No
    ``step_start`` precedes it, no ``duration_ms``.
``step_start``
    ``{"type", "name"}`` for a step of :data:`STEPS` that runs.
``step_done``
    ``{"type", "name", "stats": dict, "duration_ms": int}`` after it. A step
    that decides by itself not to act still reports ``step_done`` and says
    so in ``stats["skipped"]`` (a free-text reason): the refine steps folded
    into the merged step (``refine_merge=True``), ``materialize`` without a
    plan, ``flow_sigma_floor`` without a floor, the post-solve passes on an
    unusable solution or with an empty plan. A re-solve pass that rejected
    its re-solve adds ``stats["reverted"] = True`` and
    ``stats["revert_reason"]``; anti-overshoot reports
    ``stats["accepted"]`` (plus ``revert_reason`` on a non-converged
    re-solve) instead.
``step_error``
    ``{"type", "name", "error": repr(exc), "duration_ms": int}``; the
    exception is re-raised and no further event follows.

Two ``step_done`` events do not belong to a registry step (they appear in
neither :data:`STEPS` nor the manifest and have no ``step_start``):

* ``quality_summary`` — ``{"type", "name", "stats": {}, "duration_ms"}``,
  after the last step of :func:`run`;
* ``shunt_sanity`` — ``{"type", "name", "stats"}`` without ``duration_ms``,
  emitted by :mod:`gridstate.pipeline.orchestration` AFTER the ``final``
  event of the base run, and only when the base run converged; the trial
  runs emit nothing.

``final``
    ``{"type", "success": bool, "iterations": int}`` — last event of
    :func:`run` (of the base run when shunt sanity is on: an accepted edit
    replaces the result after ``final`` was sent).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from gridstate.api import populate_quality_summary
from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline.config import PipelineConfig
from gridstate.pipeline.context import Step, _Ctx, _no_solution
from gridstate.pipeline.steps import STEPS
from gridstate.result import SEResult
from gridstate.working import Working


# Consistency guard between config and derived (the config is used twice): the
# producer of the plans and the step executor must be gated by ONE config. An
# enabled step that did not get its plan means the configs diverged (derive
# with one cfg, run with another), not a legal input; fail early and clearly
# rather than inside the step. materialize is deliberately NOT listed: its
# plan is optional (the step skips softly with a reason — a legal input
# without an observed operating point).
_REQUIRED_DERIVED_PLANS: dict[str, tuple[str, ...]] = {
    "voltage_nominal": ("voltage_nominal",),
    "topology": ("topology_resolved",),
    "telemetry": ("telemetry_resolved", "telemetry_arg_keys"),
}


def _check_derived_consistency(
    cfg: PipelineConfig, derived: DerivedInputs, *, network_only: bool = False
) -> None:
    """Check that every enabled needs_derived step got its plan.

    Called only with ``derived is not None`` (an input without derivation is
    legal: the needs_derived steps are then skipped altogether).
    """
    missing: list[str] = []
    for step in STEPS:
        plans = _REQUIRED_DERIVED_PLANS.get(step.name)
        if not plans:
            continue
        if network_only and not step.network:
            continue
        if step.toggle is not None and not getattr(cfg, step.toggle):
            continue
        missing.extend(
            f"шаг '{step.name}' включён, derived.{attr} отсутствует"
            for attr in plans
            if getattr(derived, attr, None) is None
        )
    if missing:
        raise ValueError(
            "Рассинхрон config ↔ derived: "
            + "; ".join(missing)
            + ". Деривация планов и прогон обязаны гейтиться одним и тем же config."
        )


def _emit(on_event: Callable[[dict], None] | None, event: dict) -> None:
    if on_event is not None:
        on_event(event)


def _execute_step(step: Step, ctx: _Ctx, on_event: Callable[[dict], None] | None) -> None:
    """Run one step with the shared skip/timing/error event protocol.

    Both ``run()`` and ``prepare_network()`` go through here, so toggle/derived
    skips, duration accounting and ``step_error`` emission behave identically
    in the full pipeline and in the network-only subset.
    """
    if step.toggle is not None and not getattr(ctx.cfg, step.toggle):
        _emit(
            on_event,
            {
                "type": "step_skipped",
                "name": step.name,
                "reason": f"отключено ({step.toggle}=False)",
            },
        )
        return
    if step.needs_derived and ctx.derived is None:
        _emit(
            on_event,
            {"type": "step_skipped", "name": step.name, "reason": "no derived inputs"},
        )
        return
    _emit(on_event, {"type": "step_start", "name": step.name})
    t0 = time.monotonic()
    try:
        stats = step.fn(ctx) or {}
    except Exception as exc:
        _emit(
            on_event,
            {
                "type": "step_error",
                "name": step.name,
                "error": repr(exc),
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        raise
    _emit(
        on_event,
        {
            "type": "step_done",
            "name": step.name,
            "stats": stats,
            "duration_ms": int((time.monotonic() - t0) * 1000),
        },
    )


def _build_working(model: Any) -> Any:
    """Working layer = the gridstate-native ``Working`` container.

    Accepts either a model object with collections (``Working`` is built from
    its ``nodes``/``branches``/``measurements``/``generators``) or a ready
    :class:`~gridstate.working.Working` (its ``copy()`` is returned). The latter
    is the direct input: the caller assembles ``Working.from_arrays(...)`` from
    numpy arrays and passes it to ``run``. Either way the result is an
    independent copy: the given input stays read-only.
    """
    if isinstance(model, Working):
        # A clone, NOT a pass-through: otherwise run() would mutate the given
        # Working (pseudo measurements, V/δ). The input must stay read-only on
        # the npz input as well (a repeated run on the same object used to fail).
        return model.copy()
    return Working.from_model(model)


def _seed_warm_start(working: Any, init_state: Any) -> int:
    """Seed V/δ of the working copy from a previous ``SEResult`` (explicit warm start).

    The input is read-only, so ``run`` clones it on every call (V/δ as supplied)
    and a warm start across runs cannot come from reading an overwritten
    ``voltage_magnitude`` (as it used to). Instead the caller passes the previous
    result as ``init_state``; its V/δ (kV/rad, keyed by id) are written into the
    nodes of the working copy BEFORE preprocessing. The ``estimate`` step with
    ``init="results"`` picks them up (preprocessing does not touch V/δ).
    Returns the number of seeded nodes.

    The V/δ source is ``init_state.outputs.nodes`` (the Output table), falling
    back to ``init_state.model.nodes``.
    """
    vd: dict[int, tuple[float, float]] = {}
    out_nodes = getattr(getattr(init_state, "outputs", None), "nodes", None)
    if (
        out_nodes is not None
        and getattr(out_nodes, "size", 0) > 0
        and "voltage_magnitude" in (out_nodes.dtype.names or ())
    ):
        for i in range(len(out_nodes)):
            vd[int(out_nodes["id"][i])] = (
                float(out_nodes["voltage_magnitude"][i]),
                float(out_nodes["voltage_angle"][i]),
            )
    else:
        src = init_state.model.nodes.to_numpy()
        for r in src:
            vd[int(r["id"])] = (float(r["voltage_magnitude"]), float(r["voltage_angle"]))

    arr = working.nodes.to_numpy().copy()
    seeded = 0
    for i in range(len(arr)):
        got = vd.get(int(arr[i]["id"]))
        if got is not None:
            arr[i]["voltage_magnitude"] = got[0]
            arr[i]["voltage_angle"] = got[1]
            seeded += 1
    working.nodes.update_from_array(arr)
    return seeded


def run(
    model: Any,
    *,
    config: PipelineConfig | None = None,
    derived: DerivedInputs | None = None,
    on_event: Callable[[dict], None] | None = None,
    init_state: Any = None,
) -> SEResult:
    """Run the full SE pipeline and return an ``SEResult``. **The input is read-only.**

    ``run`` is a pure function ``(Input, config) → Output``: the input ``model``
    is NOT mutated. A working copy (the working layer) is built inside; all
    preprocessing and the solver run on it, and the result goes to ``SEResult``
    (``result.model`` is the working copy with V/δ/flows, ``result.outputs``
    holds the Output tables keyed by id). Consequences: a repeated ``run`` on
    the same ``model`` is deterministic *by construction* (no reset needed);
    edit→rerun works (edit the input, the engine only reads it); a warm start
    takes an explicit ``init_state`` with the previous result.

    **Contract change:** ``model`` used to be updated in place; read the result
    from ``result.model`` / ``result.outputs`` / ``result.v_pu`` instead of the
    passed object (it stays in its original state).

    Args:
        model: working layer (:class:`~gridstate.working.Working`) or a
            compatible carrier of the contract tables. NOT mutated.
        config: :class:`PipelineConfig`; ``None`` → library defaults.
        derived: precomputed numeric plans (:class:`~gridstate.contract.derived.
            DerivedInputs`): topology / tap changers / telemetry /
            materialization / Vnom. When given, the steps apply the ready plans
            with the contract kernels. When ``None``, the steps that need a
            numeric plan (marked ``needs_derived``) are skipped (the model must
            already carry the measurements).
        on_event: progress callback. Receives dict events ``step_start`` /
            ``step_done`` / ``step_skipped`` / ``step_error`` / ``final``.
        init_state: previous ``SEResult`` for a **warm start** (the chain
            ``run(wls)`` → ``run(ipm, init_state=res_wls)``). Its V/δ are seeded
            into the working copy before preprocessing and the estimate starts
            from them (init is forced to ``"results"``). ``None``: cold start
            from ``cfg.init``.

    Returns:
        ``SEResult`` (``ctx.result``); the result is in ``result.model`` (the
        working copy) and ``result.outputs``. The input ``model`` is unchanged.
    """
    cfg = config or PipelineConfig()
    # Re-check: a config may have been edited after construction.
    cfg.validate()

    # Working layer: gridstate-native working copy over the contract arrays. All
    # preprocessing and the solver mutate ONLY it; the input model stays
    # read-only, so a repeat is deterministic by construction (see
    # _build_working/Working) and needs no reset.
    working = _build_working(model)

    # Warm start: seed V/δ from the previous result and force init="results".
    if init_state is not None:
        seeded = _seed_warm_start(working, init_state)
        cfg = replace(cfg, init="results")
        _emit(on_event, {"type": "warm_start", "seeded_nodes": seeded})

    ctx = _Ctx(model=working, cfg=cfg)

    # The numeric plans (topology / tap changers / telemetry / materialization /
    # Vnom) come ready (``derived``): the data producer computed them outside
    # the core. The steps below apply them with the contract kernels at their
    # positions. Without plans the needs_derived steps are skipped (the model
    # must already carry the measurements).
    if derived is not None:
        _check_derived_consistency(cfg, derived)
        ctx.derived = derived

    _emit(
        on_event, {"type": "clone", "nodes": len(working.nodes), "branches": len(working.branches)}
    )

    for step in STEPS:
        _execute_step(step, ctx, on_event)

    if ctx.result is None:  # _s_estimate (no toggle) always sets the result
        raise _no_solution("quality_summary")

    # Quality summary is computed once, on the final solution: the intermediate
    # solves (estimate, anti-overshoot re-solve) run with
    # include_quality_summary=False, since on large models it costs about as
    # much as the solve itself.
    t0 = time.monotonic()
    populate_quality_summary(ctx.result, top_n=cfg.top_residuals_n)
    _emit(
        on_event,
        {
            "type": "step_done",
            "name": "quality_summary",
            "stats": {},
            "duration_ms": int((time.monotonic() - t0) * 1000),
        },
    )

    _emit(
        on_event,
        {
            "type": "final",
            "success": bool(getattr(ctx.result, "success", False)),
            "iterations": int(getattr(ctx.result, "iterations", 0)),
        },
    )
    return ctx.result


def prepare_network(
    model: Any,
    *,
    config: PipelineConfig | None = None,
    derived: DerivedInputs | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> Working:
    """Run ONLY the network derivations of the pipeline and return a ``Working``.

    Executes the ``network=True`` subset of ``STEPS`` (breaker normalization,
    Vnom, ON_LINE topology, tap changers, reactors, slack/node types, status
    cascades, generator statuses), without telemetry, pseudo measurements and
    the solver. The result is the network in the state SE solves it in.

    Purpose: materializing one network. Carrying the obtained network state
    over to the carrier model (done by an external adapter) makes the input
    consistent for SE, power flow and later computations; the derivations are
    idempotent on the materialized network (a repeated run is a no-op), so a
    following full ``run`` gives a bit-identical result.

    Args:
        model: carrier of the contract tables (as for :func:`run`). NOT mutated.
        config: :class:`PipelineConfig`; the same toggles are respected.
        derived: numeric plans; without them the needs_derived steps are skipped.
        on_event: progress callback (the same events as for :func:`run`).

    Returns:
        ``Working``: the working copy with the network derivations applied.
    """
    cfg = config or PipelineConfig()
    # Re-check: a config may have been edited after construction.
    cfg.validate()
    working: Working = _build_working(model)
    ctx = _Ctx(model=working, cfg=cfg)
    if derived is not None:
        _check_derived_consistency(cfg, derived, network_only=True)
        ctx.derived = derived

    for step in STEPS:
        if not step.network:
            continue
        _execute_step(step, ctx, on_event)
    return working
