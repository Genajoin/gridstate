"""Orchestration above a single pipeline run: the shunt sanity trial runs.

:func:`run_orchestrated` is what the contract facade
(:func:`gridstate.runtime.run`) executes: one pipeline run and, when
``config.shunt_sanity`` is on, the shunt sanity trials that compare FULL
pipeline runs of edited copies of the input (:func:`shunt_sanity_rerun`).
The trials cannot be a pipeline step: each of them is a complete run of its
own on a different network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline.config import PipelineConfig
from gridstate.pipeline.runner import run as _pipeline_run
from gridstate.result import SEResult
from gridstate.shunt_sanity import classify_shunt_candidates, edit_shunt, sum_rn2


__all__ = ["run_orchestrated", "shunt_sanity_rerun"]


def run_orchestrated(
    model: Any,
    *,
    config: PipelineConfig | None = None,
    derived: DerivedInputs | None = None,
    on_event: Callable[[dict], None] | None = None,
    init_state: Any = None,
) -> SEResult:
    """One pipeline run plus the shunt sanity trials when they are enabled.

    Args mirror :func:`gridstate.pipeline.run`; ``model`` is read-only (the
    pipeline works on a clone), which lets the trials copy it as is.
    """
    result = _pipeline_run(
        model,
        config=config,
        derived=derived,
        on_event=on_event,
        init_state=init_state,
    )
    if config is not None and config.shunt_sanity:
        result = shunt_sanity_rerun(
            base_result=result,
            model=model,
            config=config,
            derived=derived,
            on_event=on_event,
        )
    return result


def shunt_sanity_rerun(
    *,
    base_result: SEResult,
    model: Any,
    config: PipelineConfig,
    derived: DerivedInputs | None,
    on_event: Callable[[dict], None] | None,
) -> SEResult:
    """Try candidate shunts off / flipped with FULL pipeline re-runs (research).

    Validated mechanism (2026-07-06, four large models): the candidates are
    active shunts on nodes whose node V measurement disagrees with the BASE
    solution; each variant (off/flip) is applied to a copy of the input and run
    through the FULL pipeline; the gate is a drop of Σrn² (agreement with the
    model's own real measurements) larger than ``shunt_sanity_gate_drop``. The
    gate rejects false candidates.

    The comparison has to be "full run against full run": a network edit
    changes the solution, and the σ tightening of v_refine (and the v_mirror /
    bad-data plans) must be recomputed around the NEW solution; a single warm
    re-solve inside the pipeline penalises an honest edit and rejects it
    falsely.

    ``model`` is the unchanged input of the base run (the pipeline never
    mutates its input), so every trial starts from a fresh copy of it.

    Known asymmetry, kept as is: the trials are cold runs (no ``init_state``),
    while the base run may have been warm-started; with a warm base the Σrn²
    of the base and of the trials come from different starting points.

    When exactly one candidate is accepted, the joint run of the accepted
    edits is the trial run of that candidate (same input copy, same edit,
    same configuration, cold start, no events), so its result is reused
    instead of being computed a second time.
    """
    if not base_result.success:
        return base_result
    solved = base_result.model
    plan = classify_shunt_candidates(
        solved.measurements.to_numpy(),
        solved.nodes.to_numpy(),
        solved.shunts.to_numpy(),
        v_frac=float(config.shunt_sanity_v_frac),
        max_candidates=int(config.shunt_sanity_max_candidates),
    )

    def _emit(stats: dict) -> None:
        if on_event is not None:
            on_event({"type": "step_done", "name": "shunt_sanity", "stats": stats})

    if plan.empty:
        _emit({"candidates": 0, "accepted": 0, "skipped": "no-op (нет кандидатов)"})
        return base_result

    trial_cfg = replace(config, shunt_sanity=False)  # no recursion
    base_rn2 = sum_rn2(solved.measurements.to_numpy())
    gate = float(config.shunt_sanity_gate_drop)

    def _trial(edits: dict[int, str]) -> tuple[SEResult | None, float]:
        m = model.copy()
        for nid, mode in edits.items():
            if edit_shunt(m, nid, mode) == 0:
                return None, float("inf")
        res = _pipeline_run(m, config=trial_cfg, derived=derived)
        if not res.success:
            return None, float("inf")
        # The pipeline works on its own copy: Σrn² is read from res.model
        # (the trial input m keeps value=0/est=0, which would give a false zero).
        return res, sum_rn2(res.model.measurements.to_numpy())

    accepted: dict[int, str] = {}
    # Trial of the most recently accepted candidate (its result and Σrn²).
    accepted_trial: tuple[SEResult | None, float] = (None, float("inf"))
    for nid in plan.candidates:
        best_mode: str | None = None
        best_rn2 = base_rn2 - gate
        best_res: SEResult | None = None
        for mode in ("off", "flip"):
            res, rn2 = _trial({nid: mode})
            if rn2 < best_rn2:
                best_mode, best_rn2, best_res = mode, rn2, res
        if best_mode is not None:
            accepted[nid] = best_mode
            accepted_trial = (best_res, best_rn2)

    if not accepted:
        _emit(
            {
                "candidates": len(plan.candidates),
                "accepted": 0,
                "skipped": "no-op (гейт отверг всех кандидатов)",
            }
        )
        return base_result

    # One accepted candidate: _trial(accepted) would repeat its trial run.
    final_res, final_rn2 = accepted_trial if len(accepted) == 1 else _trial(accepted)
    if final_res is None or final_rn2 >= base_rn2 - gate:
        _emit(
            {
                "candidates": len(plan.candidates),
                "accepted": 0,
                "skipped": "no-op (совместное применение не прошло гейт)",
            }
        )
        return base_result
    _emit(
        {
            "candidates": len(plan.candidates),
            "accepted": len(accepted),
            "edits": {int(k): v for k, v in accepted.items()},
            "rn2_drop": float(base_rn2 - final_rn2),
        }
    )
    return final_res
