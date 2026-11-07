"""Tight zero injection of transit nodes (``transit_zero_injection_mode="tight"``).

A transit node (neither ``exist_load`` nor ``exist_gen``) injects nothing. The
soft rule gives it a zero injection prior with ``zero_inj_variance`` and skips it
where a real injection measurement exists; the tight mode puts zero P and Q on
every transit node, measured or not, with ``transit_zero_injection_variance``.
"""

from __future__ import annotations

import pytest

from gridstate.preprocessing import add_pseudo_measurements
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    OBJ_BRANCH,
    OBJ_NODE,
)


def _model(*, p_inj_at_2: bool = False):
    """Chain 1 - 2 - 3: node 1 generates, node 2 is transit, node 3 has load.

    Real measurements: P flow on both branches and, optionally, a P injection
    at the transit node 2 (no Q injection there).
    """
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid, gen, load in ((1, True, False), (2, False, False), (3, False, True)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 220.0,
                "voltage_magnitude": 220.0,
                "status": True,
                "node_type": int(NodeType.SLACK if nid == 1 else NodeType.PQ),
                "exist_gen": gen,
                "exist_load": load,
                "generation_p": 50.0 if gen else 0.0,
                "load_p": 50.0 if load else 0.0,
                # an input value on the transit node: the tight zero ignores it
                "load_q": 3.0 if nid == 2 else 0.0,
            }
        )
    for bid, f, t in ((12, 1, 2), (23, 2, 3)):
        m.branches.add(
            {
                "id": bid,
                "from_node": f,
                "to_node": t,
                "status": True,
                "branch_type": 0,
                "tap_ratio": 1.0,
                "resistance": 1.0,
                "reactance": 10.0,
            }
        )
    rows = [
        (1, OBJ_BRANCH, 12, KIND_POWER_P, 50.0),
        (2, OBJ_BRANCH, 23, KIND_POWER_P, 49.0),
    ]
    if p_inj_at_2:
        rows.append((3, OBJ_NODE, 2, KIND_POWER_INJECTION_P, 0.5))
    for mid, ot, oid, mt, value in rows:
        m.measurements.add(
            {
                "id": mid,
                "object_type": ot,
                "object_id": oid,
                "measurement_type": mt,
                "branch_side": 0 if ot == OBJ_BRANCH else -1,
                "value": value,
                "variance": 4.0,
                "weight": 0.25,
                "status": True,
            }
        )
    return m


def _pseudo_injections(m, node_id: int) -> dict[int, tuple[float, float]]:
    """``{kind: (value, variance)}`` of the pseudo injection rows of ``node_id``."""
    arr = m.measurements.to_numpy()
    sel = (
        arr["is_pseudo"].astype(bool)
        & (arr["object_type"] == OBJ_NODE)
        & (arr["object_id"] == node_id)
    )
    out: dict[int, tuple[float, float]] = {}
    for r in arr[sel]:
        kind = int(r["measurement_type"])
        if kind in (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q):
            assert kind not in out, "one pseudo row per kind"
            out[kind] = (float(r["value"]), float(r["variance"]))
    return out


def test_soft_is_the_default_rule():
    m = _model()
    stats = add_pseudo_measurements(m)
    assert "transit_zero_tight" not in stats
    rows = _pseudo_injections(m, 2)
    assert rows[KIND_POWER_INJECTION_P] == (0.0, 100.0)
    assert rows[KIND_POWER_INJECTION_Q] == (-3.0, 100.0)


def test_soft_skips_a_measured_transit_node():
    m = _model(p_inj_at_2=True)
    add_pseudo_measurements(m)
    assert _pseudo_injections(m, 2) == {}


def test_tight_zero_on_every_transit_node():
    m = _model(p_inj_at_2=True)
    stats = add_pseudo_measurements(
        m, transit_zero_injection_mode="tight", transit_zero_injection_variance=1e-2
    )
    assert stats["transit_zero_tight"] == 1
    rows = _pseudo_injections(m, 2)
    assert rows[KIND_POWER_INJECTION_P] == (0.0, 1e-2)
    assert rows[KIND_POWER_INJECTION_Q] == (0.0, 1e-2)


def test_tight_leaves_other_nodes_alone():
    soft, tight = _model(), _model()
    add_pseudo_measurements(soft)
    add_pseudo_measurements(tight, transit_zero_injection_mode="tight")
    for nid in (1, 3):
        assert _pseudo_injections(tight, nid) == _pseudo_injections(soft, nid)


def test_tight_needs_the_node_markup():
    """Without exist_load/exist_gen the fallback transit criterion stays soft."""
    soft, tight = _model(), _model()
    for m in (soft, tight):
        arr = m.nodes.to_numpy()
        arr["exist_gen"] = False
        arr["exist_load"] = False
        m.nodes.update_from_array(arr)
    add_pseudo_measurements(soft)
    stats = add_pseudo_measurements(tight, transit_zero_injection_mode="tight")
    assert stats["transit_zero_tight"] == 0
    assert _pseudo_injections(tight, 2) == _pseudo_injections(soft, 2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"transit_zero_injection_mode": "hard"}, "transit_zero_injection_mode"),
        ({"transit_zero_injection_variance": 0.0}, "transit_zero_injection_variance"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        add_pseudo_measurements(_model(), **kwargs)


def test_pipeline_config_fields():
    from gridstate.pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.transit_zero_injection_mode == "soft"
    assert cfg.transit_zero_injection_variance == 1e-2
    with pytest.raises(ValueError, match="transit_zero_injection_mode"):
        PipelineConfig(transit_zero_injection_mode="hard")
