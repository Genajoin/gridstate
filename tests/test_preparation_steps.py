"""Plan-driven preparation steps of the pipeline, called directly on a context.

The steps that read a :class:`~gridstate.contract.derived.DerivedInputs` plan
must fail loudly when the plan is absent and apply it otherwise. The IEEE and
synthetic runs elsewhere carry no plans, so these wrappers (and the observed
operating point kernel behind ``materialize``) are exercised here on small
hand-made models.
"""

from __future__ import annotations

import pytest

from gridstate.constants import NodeType
from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline.config import PipelineConfig
from gridstate.pipeline.context import _Ctx
from gridstate.pipeline.preparation import (
    _s_materialize,
    _s_relax_generation_min_to_injection,
    _s_release_load_only_injections,
    _s_rpn,
    _s_telemetry,
    _s_topology,
    _s_voltage_nominal,
    _s_widen_generation_only_injections,
)
from gridstate.working import Working


def _nodes_model() -> Working:
    """Area 1: node 1 (slack, generator), nodes 2-4 loads; node 4 is switched off."""
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "area_id": 1,
            "voltage_nominal": 110.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
            "exist_gen": 1,
            "generation_p": 5.0,
            "generation_p_max": 300.0,
            "generation_q_max": 100.0,
            "generation_q_min": -100.0,
        }
    )
    for nid, p_max, status in ((2, 100.0, True), (3, 50.0, True), (4, 80.0, False)):
        m.nodes.add(
            {
                "id": nid,
                "area_id": 1,
                "voltage_nominal": 0.0 if nid == 3 else 110.0,
                "status": status,
                "node_type": int(NodeType.PQ),
                "exist_load": 1,
                "load_p": 1.0,
                "load_p_max": p_max,
                "load_q_max": p_max / 2.0,
            }
        )
    return m


def _ctx(model: Working, derived: DerivedInputs | None) -> _Ctx:
    return _Ctx(model=model, cfg=PipelineConfig(), derived=derived)


@pytest.mark.parametrize(
    ("step", "plan_name"),
    [
        (_s_voltage_nominal, "voltage_nominal"),
        (_s_topology, "topology"),
        (_s_telemetry, "telemetry"),
        (_s_release_load_only_injections, "release_load_only_injections"),
        (_s_relax_generation_min_to_injection, "relax_generation_min_to_injection"),
        (_s_widen_generation_only_injections, "widen_generation_only_injections"),
    ],
)
@pytest.mark.parametrize("derived", [None, DerivedInputs()])
def test_plan_steps_fail_without_their_plan(step, plan_name, derived) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=plan_name):
        step(_ctx(_nodes_model(), derived))


def test_voltage_nominal_step_applies_the_plan() -> None:
    m = _nodes_model()
    assert _s_voltage_nominal(_ctx(m, DerivedInputs(voltage_nominal={3: 35.0}))) == {}
    row = m.nodes.get_by_id(3)
    assert row is not None
    assert row.voltage_nominal == 35.0


def test_rpn_step_reads_the_tap_steps_table() -> None:
    m = _nodes_model()
    assert _s_rpn(_ctx(m, None)) == {"applied": 0, "shunt_recalc": 0}


def test_injection_steps_with_an_empty_telemetry_plan_change_nothing() -> None:
    m = _nodes_model()
    before = m.nodes.to_numpy().copy()
    plan = DerivedInputs(telemetry_resolved={}, telemetry_arg_keys=[])
    ctx = _ctx(m, plan)
    stats = _s_release_load_only_injections(ctx)
    assert ctx.released_injections == {"P": frozenset(), "Q": frozenset()}
    assert all(v == 0 for v in stats.values() if isinstance(v, int))
    _s_relax_generation_min_to_injection(ctx)
    _s_widen_generation_only_injections(ctx)
    after = m.nodes.to_numpy()
    for name in ("load_p", "generation_p", "generation_p_min"):
        assert (after[name] == before[name]).all()


def test_materialize_without_plan_is_skipped() -> None:
    m = _nodes_model()
    out = _s_materialize(_ctx(m, None))
    assert "skipped" in out
    assert _s_materialize(_ctx(m, DerivedInputs()))["skipped"] == out["skipped"]


def test_materialize_writes_observed_values_and_fills_by_area_ratio() -> None:
    """Observed nodes get the value verbatim. An unobserved active load node gets
    ``k * load_max`` with ``k`` the area median of ``observed / load_max``:
    node 2 observed 40 of 100 -> k = 0.4 -> node 3 = 0.4 * 50 = 20 MW,
    Q: node 2 observed 10 of 50 -> k = 0.2 -> node 3 = 0.2 * 25 = 5 Mvar.
    The switched-off node 4 is left alone. Generation is written verbatim only
    (no fill by default)."""
    m = _nodes_model()
    obs = {
        "load_p": {2: 40.0},
        "load_q": {2: 10.0},
        "gen_p": {1: 120.0},
        "gen_q": {},
    }
    assert _s_materialize(_ctx(m, DerivedInputs(materialize_obs=obs))) == {}
    nodes = {int(r["id"]): r for r in m.nodes.to_numpy()}
    assert nodes[2]["load_p"] == 40.0
    assert nodes[3]["load_p"] == pytest.approx(20.0)
    assert nodes[2]["load_q"] == 10.0
    assert nodes[3]["load_q"] == pytest.approx(5.0)
    assert nodes[4]["load_p"] == 1.0  # switched off: untouched
    assert nodes[1]["generation_p"] == 120.0
    assert nodes[1]["generation_q"] == 0.0  # not observed, generation is not filled


def test_topology_step_applies_the_status_plan() -> None:
    """``(object_kind, id, status, skip_reason)`` items switch objects on/off; items without
    a value and unknown objects are counted and skipped."""
    m = _nodes_model()
    plan = DerivedInputs(
        topology_resolved=[
            ("node", 2, False, None),
            ("node", 4, True, None),
            ("node", 3, None, "no_value"),
            ("node", 99, True, None),
        ]
    )
    stats = _s_topology(_ctx(m, plan))
    assert stats["applied_off"] == 1
    assert stats["applied_on"] == 1
    assert stats["skipped_no_value"] == 1
    assert stats["skipped_no_object"] == 1
    assert stats["total_specs"] == 4
    status = {int(r["id"]): bool(r["status"]) for r in m.nodes.to_numpy()}
    assert status == {1: True, 2: False, 3: True, 4: True}
