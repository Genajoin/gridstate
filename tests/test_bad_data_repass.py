"""Unit tests for ``gridstate.bad_data_repass`` + the pipeline step.

Classification is tested on synthetic arrays (no solve): every branch of the
mechanism -- sign flip, reject of a broken zero, pair immunity, Qinj damping,
domain-coverage guard -- is isolated deterministically. The
``bad_data_repass`` step in ``run()`` is checked end to end: on a clean model the
plan is empty -> no-op (solution bit-identical to the step switched off, the
second solve is not called).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gridstate.bad_data_repass import (
    BadDataPlan,
    apply_bad_data_plan,
    classify_bad_data,
)
from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig, run
from gridstate.pipeline.context import _Ctx
from gridstate.pipeline.passes import _s_bad_data_repass
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_idempotent import _make_model_with_reactor


# ---------------------------------------------------------------------------
# Synthetic builder: Working with branch 1-2 and measurements with given (z, h, sigma).
# ---------------------------------------------------------------------------


def _build_working() -> Working:
    m = Working.empty()
    for nid, ntype in ((1, NodeType.SLACK), (2, NodeType.PQ)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
            }
        )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 1.0,
            "reactance": 10.0,
            "tap_ratio": 1.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    return m


def _add_meas(
    m: Working,
    mid: int,
    kind: int,
    obj_id: int,
    *,
    z: float,
    h: float,
    sigma: float,
    side: int = 0,
    obj_type: int | None = None,
) -> None:
    if obj_type is None:
        obj_type = OBJ_BRANCH if kind == KIND_POWER_P else OBJ_NODE
    m.measurements.add(
        {
            "id": mid,
            "object_type": obj_type,
            "object_id": obj_id,
            "measurement_type": kind,
            "branch_side": side,
            "value": z,
            "variance": sigma * sigma,
            "status": True,
            "is_pseudo": False,
            "estimated_si": h,
        }
    )


def _classify(m: Working) -> BadDataPlan:
    return classify_bad_data(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        threshold=10.0,
        sigma_cap=30.0,
        flip_ratio=0.33,
    )


def _add_domain_anchors(m: Working, next_id: int = 900) -> None:
    """Consistent Pinj/Qinj at both nodes -- domain coverage, not candidates."""
    for nid in (1, 2):
        _add_meas(m, next_id, KIND_POWER_INJECTION_P, nid, z=10.0, h=10.0, sigma=5.0)
        _add_meas(m, next_id + 1, KIND_POWER_INJECTION_Q, nid, z=5.0, h=5.0, sigma=5.0)
        next_id += 2


# ---------------------------------------------------------------------------
# classify_bad_data
# ---------------------------------------------------------------------------


def test_flip_detects_sign_flipped_flow():
    """A sign-flipped P-flow (z ~ -h) is cured by a flip, not a reject."""
    m = _build_working()
    _add_domain_anchors(m)
    _add_meas(m, 1, KIND_POWER_P, 100, z=-100.0, h=98.0, sigma=5.0)

    plan = _classify(m)

    assert plan.flip_ids == frozenset({1})
    assert plan.reject_ids == frozenset()
    assert plan.damp_ids == frozenset()


def test_reject_dead_zero_flow():
    """A broken zero (z=0 at h=200) is rejected (flipping a zero is meaningless)."""
    m = _build_working()
    _add_domain_anchors(m)
    _add_meas(m, 1, KIND_POWER_P, 100, z=0.0, h=200.0, sigma=10.0)

    plan = _classify(m)

    assert plan.reject_ids == frozenset({1})
    assert plan.flip_ids == frozenset()


def test_pair_immunity_protects_consistent_pair():
    """A consistent branch pair with large residuals is immune (anti-circular)."""
    m = _build_working()
    _add_meas(m, 1, KIND_POWER_P, 100, z=600.0, h=100.0, sigma=20.0, side=0)
    _add_meas(m, 2, KIND_POWER_P, 100, z=-590.0, h=-95.0, sigma=20.0, side=1)

    plan = _classify(m)

    assert plan.empty
    assert plan.n_candidates == 2
    assert plan.n_immune == 2


def test_zero_pair_gets_no_immunity():
    """A zero pair (z=0/0 at large h) is NOT immune: z=0 carries no evidence."""
    m = _build_working()
    _add_domain_anchors(m)
    _add_meas(m, 1, KIND_POWER_P, 100, z=0.0, h=200.0, sigma=10.0, side=0)
    _add_meas(m, 2, KIND_POWER_P, 100, z=0.0, h=-195.0, sigma=10.0, side=1)

    plan = _classify(m)

    assert plan.n_immune == 0
    assert plan.reject_ids == frozenset({1, 2})


def test_qinj_damped_never_rejected():
    """A monster Qinj is damped (sigma*k), not disabled (blind-edge case)."""
    m = _build_working()
    _add_domain_anchors(m)
    _add_meas(m, 1, KIND_POWER_INJECTION_Q, 2, z=1400.0, h=0.0, sigma=70.0)

    plan = _classify(m)

    assert plan.damp_ids == frozenset({1})
    assert plan.reject_ids == frozenset()


def test_coverage_guard_restores_last_domain_measure():
    """A node does not lose its last real P-domain measurement -- the reject is cancelled."""
    m = _build_working()
    # The only P measurement of node 2 is a monster Pinj; the Q domain is covered.
    _add_meas(m, 1, KIND_POWER_INJECTION_P, 2, z=500.0, h=0.0, sigma=10.0)
    _add_meas(m, 2, KIND_POWER_INJECTION_Q, 2, z=5.0, h=5.0, sigma=5.0)

    plan = _classify(m)

    assert plan.reject_ids == frozenset()
    assert plan.n_restored == 1


def test_clean_model_yields_empty_plan():
    """Consistent measurements -> no candidates, the plan is empty."""
    m = _build_working()
    _add_domain_anchors(m)
    _add_meas(m, 1, KIND_POWER_P, 100, z=50.0, h=49.0, sigma=5.0)

    plan = _classify(m)

    assert plan.empty
    assert plan.n_candidates == 0


def test_pseudo_and_voltage_measures_ignored():
    """Pseudo-measurements and V measurements do not take part in detection."""
    m = _build_working()
    _add_domain_anchors(m)
    # A coarse V measurement and a coarse pseudo-Pinj -- both outside the DETECTABLE real set.
    _add_meas(m, 1, KIND_VOLTAGE, 2, z=200.0, h=110.0, sigma=1.0, obj_type=OBJ_NODE)
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 999.0,
            "variance": 1.0,
            "status": True,
            "is_pseudo": True,
            "estimated_si": 0.0,
        }
    )

    plan = _classify(m)

    assert plan.empty


# ---------------------------------------------------------------------------
# apply_bad_data_plan
# ---------------------------------------------------------------------------


def test_apply_plan_mutates_measurements():
    m = _build_working()
    _add_meas(m, 1, KIND_POWER_P, 100, z=-100.0, h=98.0, sigma=5.0)
    _add_meas(m, 2, KIND_POWER_P, 100, z=0.0, h=200.0, sigma=10.0, side=1)
    _add_meas(m, 3, KIND_POWER_INJECTION_Q, 2, z=1400.0, h=0.0, sigma=70.0)
    plan = BadDataPlan(
        flip_ids=frozenset({1}),
        reject_ids=frozenset({2}),
        damp_ids=frozenset({3}),
        n_candidates=3,
        n_immune=0,
        n_restored=0,
    )

    stats = apply_bad_data_plan(m, plan, damp_factor=5.0)

    me = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert float(me[1]["value"]) == 100.0  # sign flip
    assert bool(me[2]["status"]) is False  # deactivation
    assert bool(me[3]["status"]) is True  # Qinj stayed active
    assert float(me[3]["variance"]) == 70.0 * 70.0 * 25.0  # σ×5 → variance×25
    assert stats == {
        "flips": 1,
        "rejects": 1,
        "damped": 1,
        "candidates": 3,
        "immune": 0,
        "restored": 0,
    }


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def test_step_skips_on_unusable_solution():
    """completed=False -> classification is not called (residuals are unreliable)."""
    ctx = _Ctx(model=None, cfg=PipelineConfig(bad_data=True))
    ctx.result = SimpleNamespace(success=False)

    stats = _s_bad_data_repass(ctx)

    assert "skipped" in stats


def test_run_noop_on_clean_model_bit_exact():
    """On a consistent model (no shunt: flat state = exact solution)
    the plan is empty: solution bit-identical to bad_data=False, no re-solve."""
    m = _make_model_with_reactor(susceptance_uS=0.0)
    r_off = run(m, config=PipelineConfig(algorithm="wls"))

    events: list[dict] = []
    r_on = run(
        m,
        config=PipelineConfig(algorithm="wls", bad_data=True),
        on_event=events.append,
    )

    step_events = [
        e for e in events if e.get("type") == "step_done" and e.get("name") == "bad_data_repass"
    ]
    assert step_events and "skipped" in step_events[0]["stats"]
    a = r_off.model.nodes.to_numpy()
    b = r_on.model.nodes.to_numpy()
    assert np.array_equal(a["voltage_magnitude"], b["voltage_magnitude"])
    assert np.array_equal(a["voltage_angle"], b["voltage_angle"])
    assert r_on.iterations == r_off.iterations


def test_run_repass_on_conflicting_model():
    """A model with a 605 MVAr shunt reactor against zero Qinj measurements: the conflict is detected,
    Qinj are damped (not rejected), the re-solve succeeds."""
    m = _make_model_with_reactor()

    events: list[dict] = []
    r = run(
        m,
        config=PipelineConfig(algorithm="wls", bad_data=True),
        on_event=events.append,
    )

    step_events = [
        e for e in events if e.get("type") == "step_done" and e.get("name") == "bad_data_repass"
    ]
    assert step_events
    stats = step_events[0]["stats"]
    assert stats["candidates"] > 0
    assert stats["damped"] > 0
    assert stats["success"] is True
    assert r.success
