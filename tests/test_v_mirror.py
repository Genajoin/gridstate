"""Unit tests for ``gridstate.v_mirror`` + the pipeline step.

Classification on synthetic data (with ``voltage_magnitude`` set to the "first-pass
solution"): a blind node (no real telemetry, flat pseudo-V) systematically
below the bound of ITS OWN class -> goes into the plan with value ``pu*Vnom``. The gates cut:
a bound of another class (autotransformer), a node already at level (lift), a junk bound
(max_pu_dev), a non-flat prior. The ``v_mirror`` step in ``run()`` is checked end to end on
a no-op (a model without blind nodes -> solution bit-identical).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig, run
from gridstate.pipeline.context import _Ctx
from gridstate.pipeline.passes import _s_v_mirror
from gridstate.v_mirror import (
    VMirrorPlan,
    apply_v_mirror_plan,
    classify_v_mirror,
)
from gridstate.working import Working
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_idempotent import _make_model_with_reactor


# ---------------------------------------------------------------------------
# Synthetic builder: bound node (real V) + blind node (flat pseudo-V)
# ---------------------------------------------------------------------------


def _build(
    *,
    boundary_vn: float = 110.0,
    boundary_vmag: float = 121.0,  # bound solved at pu 1.10
    blind_vn: float = 110.0,
    blind_vmag: float = 110.9,  # blind node solved at pu 1.008 -> lift 0.092
    blind_pseudo_value: float = 110.0,  # flat placeholder == Vnom
    branch_type: int = int(BranchType.LINE),
    tap_ratio: float = 1.0,
) -> Working:
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": boundary_vn,
            "voltage_magnitude": boundary_vmag,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": blind_vn,
            "voltage_magnitude": blind_vmag,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 1.0,
            "reactance": 10.0,
            "tap_ratio": tap_ratio,
            "status": True,
            "branch_type": branch_type,
        }
    )
    # real V measurement at the bound (node 1) -> node 1 measured, node 2 blind
    m.measurements.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": boundary_vmag,
            "variance": 1.0,
            "status": True,
            "is_pseudo": False,
        }
    )
    # flat pseudo-V at the blind node 2
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": blind_pseudo_value,
            "variance": (0.05 * blind_vn) ** 2,
            "status": True,
            "is_pseudo": True,
        }
    )
    return m


def _classify(m: Working, *, max_pu_dev: float = 0.25, min_lift: float = 0.01) -> VMirrorPlan:
    return classify_v_mirror(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        m.nodes.to_numpy(),
        max_pu_dev=max_pu_dev,
        min_lift=min_lift,
    )


# ---------------------------------------------------------------------------
# classify_v_mirror
# ---------------------------------------------------------------------------


def test_blind_node_lifted_to_boundary():
    """A blind node below the bound of its class -> value = pu*Vnom."""
    plan = _classify(_build())
    assert plan.n_clusters == 1
    assert len(plan.new_values) == 1
    nid, val = plan.new_values[0]
    assert nid == 2
    assert abs(val - 121.0) < 1e-6  # bound pu 1.10 * Vnom 110


def test_boundary_other_class_ignored():
    """A bound of another class (blind 110 behind an autotransformer from 220) -> left alone."""
    plan = _classify(_build(boundary_vn=220.0, boundary_vmag=242.0))
    assert plan.empty


def test_cross_at_mirrors_via_transformer():
    """A blind 110 behind an autotransformer (branch_type=1) from a 220 bound -> cross_at=True
    carries over the pu of the solved trafo bound (pu is invariant to an ideal tap)."""
    m = _build(
        boundary_vn=220.0,
        boundary_vmag=242.0,  # pu 1.10
        branch_type=int(BranchType.TRANSFORMER),
        tap_ratio=2.0,
    )
    # cross_at=False (default) -> no same-class bound -> empty.
    assert classify_v_mirror(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        m.nodes.to_numpy(),
        max_pu_dev=0.25,
        min_lift=0.01,
    ).empty
    # cross_at=True -> pu(220 bound)=1.10 -> value = 1.10*110 = 121.
    plan = classify_v_mirror(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        m.nodes.to_numpy(),
        max_pu_dev=0.25,
        min_lift=0.01,
        cross_at=True,
    )
    assert plan.n_clusters == 1
    nid, val = plan.new_values[0]
    assert nid == 2
    assert abs(val - 121.0) < 1e-6


def test_cross_at_skips_below_110kv():
    """On the LV side (blind 10 kV behind an autotransformer from 110) cross_at does NOT fire --
    pu invariance across a generator step-up is grossly violated (threshold >= 110 kV)."""
    m = _build(
        boundary_vn=110.0,
        boundary_vmag=121.0,
        blind_vn=10.0,
        blind_vmag=10.05,
        blind_pseudo_value=10.0,
        branch_type=int(BranchType.TRANSFORMER),
        tap_ratio=11.0,
    )
    plan = classify_v_mirror(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        m.nodes.to_numpy(),
        max_pu_dev=0.25,
        min_lift=0.01,
        cross_at=True,
    )
    assert plan.empty  # blind_vn=10 < 110 -> the cross-autotransformer fallback is not activated


def test_node_at_level_ignored():
    """A node already at the bound level (lift <= min_lift) -> left alone (no push)."""
    plan = _classify(_build(blind_vmag=120.9))  # pu 1.099 ~ bound 1.10
    assert plan.empty


def test_non_flat_pseudo_ignored():
    """A pseudo-V with a meaningful operating point (value != Vnom) is anchored, left alone."""
    plan = _classify(_build(blind_pseudo_value=115.0))
    assert plan.empty


def test_measured_node_not_blind():
    """A node with real V (not pseudo) is observed, not blind -> outside the plan."""
    m = _build()
    arr = m.measurements.to_numpy()
    arr["is_pseudo"][arr["id"] == 2] = False  # node 2 now has real V
    m.measurements.update_from_array(arr)
    plan = _classify(m)
    assert plan.empty


def test_max_pu_dev_gate():
    """A wild bound (pu outside [1 +/- max_pu_dev]) -> discarded."""
    plan = _classify(_build(boundary_vmag=176.0), max_pu_dev=0.25)  # pu 1.6
    assert plan.empty


def test_lift_gate_just_below():
    """A node slightly below the lift threshold (pu 1.091 at bound 1.10, lift 0.009 < 0.01)
    -> left alone."""
    plan = _classify(_build(blind_vmag=120.0), min_lift=0.01)
    assert plan.empty


# ---------------------------------------------------------------------------
# apply_v_mirror_plan
# ---------------------------------------------------------------------------


def test_apply_sets_pseudo_value():
    m = _build()
    plan = VMirrorPlan(new_values=((2, 121.0),), n_clusters=1)
    stats = apply_v_mirror_plan(m, plan)
    me = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert float(me[2]["value"]) == 121.0  # pseudo-V moved
    assert float(me[1]["value"]) == 121.0  # real V of the bound untouched (coincidence)
    assert stats == {"clusters": 1, "nodes": 1}


def test_apply_empty_noop():
    m = _build()
    plan = VMirrorPlan(new_values=(), n_clusters=0)
    stats = apply_v_mirror_plan(m, plan)
    assert stats == {"clusters": 0, "nodes": 0}
    me = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert float(me[2]["value"]) == 110.0  # untouched


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


def test_step_skips_on_unusable_solution():
    """success=False -> classification is not called (the bound level is unreliable)."""
    ctx = _Ctx(model=None, cfg=PipelineConfig(v_mirror=True))
    ctx.result = SimpleNamespace(success=False)
    stats = _s_v_mirror(ctx)
    assert "skipped" in stats


def test_run_noop_without_blind_nodes_bit_exact():
    """A model where all active nodes are observable (no blind clusters): the
    v_mirror plan is empty -> solution bit-identical to v_mirror=False, no re-solve."""
    m = _make_model_with_reactor(susceptance_uS=0.0)
    r_off = run(m, config=PipelineConfig(algorithm="wls"))

    events: list[dict] = []
    r_on = run(
        m,
        config=PipelineConfig(algorithm="wls", v_mirror=True),
        on_event=events.append,
    )

    step_events = [
        e for e in events if e.get("type") == "step_done" and e.get("name") == "v_mirror"
    ]
    assert step_events and "skipped" in step_events[0]["stats"]
    a = r_off.model.nodes.to_numpy()
    b = r_on.model.nodes.to_numpy()
    assert np.array_equal(a["voltage_magnitude"], b["voltage_magnitude"])
    assert np.array_equal(a["voltage_angle"], b["voltage_angle"])
    assert r_on.iterations == r_off.iterations
