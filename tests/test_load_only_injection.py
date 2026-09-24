"""Partial node injections (gridstate.telemetry.load_only_injection).

A node injection assembled from the load component alone (PN without PG, QN
without QG) states zero generation. On a node with generation it is partial and
must be released; the pseudo-injection prior then spans the generation range.
"""

from __future__ import annotations

import math

from gridstate.constants import FilterFlag, NodeType
from gridstate.preprocessing.pseudo_measurements import add_pseudo_measurements
from gridstate.telemetry.load_only_injection import (
    load_only_injection_nodes,
    release_load_only_injections,
)
from gridstate.telemetry.quality import QUALITY_BAD
from gridstate.working import Working


GOOD = 0


def _resolved(**comp: tuple[int, str, float | None, int]):
    """``name=(node, kind, value, quality)`` -> ``(resolved, arg_keys)``."""
    resolved = {}
    keys = []
    for node, kind, value, quality in comp.values():
        resolved[(node, kind)] = (value, 1, "g", quality)
        keys.append((node, kind))
    return resolved, keys


def test_load_only_components_are_detected():
    resolved, keys = _resolved(
        a=(1, "PN", 40.0, GOOD),  # load only -> P partial
        b=(1, "QN", 29.0, GOOD),
        c=(2, "PN", 10.0, GOOD),  # load + generator formula -> complete
        d=(2, "PG_G1", 500.0, GOOD),
        e=(3, "PG", 100.0, GOOD),  # generation only -> not load-only
        f=(4, "PN", 5.0, GOOD),  # generation reported BAD -> still load-only
        g=(4, "PG", 300.0, QUALITY_BAD),
        h=(5, "PN", None, GOOD),  # no value -> no component at all
    )
    out = load_only_injection_nodes(resolved, keys)
    assert out["P"] == {1, 4}
    assert out["Q"] == {1}


def _model(*, p_range: tuple[float, float], q_range: tuple[float, float] = (-100.0, 100.0)):
    """Slack(1) — unit transformer — unit node(2) with generation and auxiliary load."""
    m = Working.empty()
    for nid, ntype in [(1, NodeType.SLACK), (2, NodeType.PQ)]:
        m.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": 20.0 if nid == 2 else 750.0,
                "voltage_magnitude": 20.0 if nid == 2 else 750.0,
                "status": True,
                "node_type": int(ntype),
                "exist_gen": nid == 2,
                "exist_load": nid == 2,
                "generation_p_min": p_range[0] if nid == 2 else 0.0,
                "generation_p_max": p_range[1] if nid == 2 else 0.0,
                "generation_q_min": q_range[0] if nid == 2 else 0.0,
                "generation_q_max": q_range[1] if nid == 2 else 0.0,
                "load_p": 40.0 if nid == 2 else 0.0,
                "load_q": 29.0 if nid == 2 else 0.0,
            }
        )
    m.branches.add(
        {
            "id": 12,
            "name": "unit-transformer",
            "from_node": 1,
            "to_node": 2,
            "status": True,
            "branch_type": 1,
            "tap_ratio": 1.0,
            "resistance": 0.1,
            "reactance": 10.0,
        }
    )
    for mid, mt, value in [(1, 4, -40.0), (2, 5, -29.0)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": 0,
                "object_id": 2,
                "measurement_type": mt,
                "branch_side": -1,
                "value": value,
                "variance": 4.0,
                "weight": 0.25,
                "status": True,
            }
        )
    return m


def _meas(m: Working, mt: int):
    arr = m.measurements.to_numpy()
    sel = (arr["object_id"] == 2) & (arr["measurement_type"] == mt)
    return arr[sel]


def test_releases_load_only_injection_on_generating_node():
    m = _model(p_range=(0.0, 525.0))
    stats, released = release_load_only_injections(m, {"P": frozenset({2}), "Q": frozenset({2})})
    assert stats == {"released_p": 1, "released_q": 1}
    assert released == {"P": frozenset({2}), "Q": frozenset({2})}
    for mt in (4, 5):
        row = _meas(m, mt)[0]
        assert not row["status"]
        assert row["filter_flag"] == int(FilterFlag.LOAD_ONLY_INJECTION)


def test_releases_on_pumped_storage_ranges():
    """Pumped storage: pumping-only ``[-200, 0]`` and reversible ranges count."""
    for p_range in ((-200.0, 0.0), (-1200.0, 1200.0)):
        m = _model(p_range=p_range)
        stats, _ = release_load_only_injections(m, {"P": frozenset({2}), "Q": frozenset()})
        assert stats == {"released_p": 1, "released_q": 0}


def test_keeps_injection_without_declared_generation_range():
    """``exist_gen`` with an unset [0, 0] range (equivalent, compensator): kept."""
    m = _model(p_range=(0.0, 0.0))
    stats, _ = release_load_only_injections(m, {"P": frozenset({2}), "Q": frozenset({2})})
    assert stats == {"released_p": 0, "released_q": 0}
    assert _meas(m, 4)[0]["status"]
    assert _meas(m, 5)[0]["status"]


def test_keeps_injection_on_node_without_generation():
    m = _model(p_range=(0.0, 525.0))
    nodes = m.nodes.to_numpy().copy()
    nodes["exist_gen"][nodes["id"] == 2] = False
    m.nodes.update_from_array(nodes)
    stats, _ = release_load_only_injections(m, {"P": frozenset({2}), "Q": frozenset({2})})
    assert stats == {"released_p": 0, "released_q": 0}


def test_pseudo_prior_spans_generation_range():
    """Released node: P_inj prior sigma = half the generation range, not the load prior."""
    m = _model(p_range=(0.0, 525.0))
    _, released = release_load_only_injections(m, {"P": frozenset({2}), "Q": frozenset({2})})
    add_pseudo_measurements(
        m,
        unmeasured_gen_p_nodes=released["P"],
        unmeasured_gen_q_nodes=released["Q"],
    )
    arr = m.measurements.to_numpy()
    pseudo = arr[arr["is_pseudo"].astype(bool) & (arr["object_id"] == 2)]
    var = {int(r["measurement_type"]): float(r["variance"]) for r in pseudo}
    assert math.isclose(var[4], 262.5**2)
    assert math.isclose(var[5], 100.0**2)


# --- generation-only injections on nodes with load ---------------------------


def test_generation_only_components_are_detected():
    from gridstate.telemetry.load_only_injection import generation_only_injection_nodes

    resolved, keys = _resolved(
        a=(1, "PG_G0", 0.0, GOOD),  # generator only -> P gen-only
        b=(1, "QG_G0", 0.0, GOOD),
        c=(2, "PG", 100.0, GOOD),  # generation + load -> complete
        d=(2, "PN", 10.0, GOOD),
        e=(3, "PN", 5.0, GOOD),  # load only -> not gen-only
    )
    out = generation_only_injection_nodes(resolved, keys)
    assert out["P"] == {1}
    assert out["Q"] == {1}


def _load_node_model(*, load_p: tuple[float, float], load_q: tuple[float, float] = (0.0, 0.0)):
    """Unit node(2) with generation, a load box and a gen-only injection +100 / +20."""
    m = _model(p_range=(0.0, 525.0))
    nodes = m.nodes.to_numpy().copy()
    sel = nodes["id"] == 2
    nodes["load_p_min"][sel], nodes["load_p_max"][sel] = load_p
    nodes["load_q_min"][sel], nodes["load_q_max"][sel] = load_q
    m.nodes.update_from_array(nodes)
    meas = m.measurements.to_numpy().copy()
    meas["value"][meas["measurement_type"] == 4] = 100.0
    meas["value"][meas["measurement_type"] == 5] = 20.0
    m.measurements.update_from_array(meas)
    return m


def test_widens_generation_only_injection_by_load_box():
    from gridstate.telemetry.load_only_injection import widen_generation_only_injections

    m = _load_node_model(load_p=(0.0, 60.0), load_q=(10.0, 30.0))
    stats = widen_generation_only_injections(m, {"P": frozenset({2}), "Q": frozenset({2})})
    assert stats == {"widened_p": 1, "widened_q": 1}
    p = _meas(m, 4)[0]
    q = _meas(m, 5)[0]
    assert math.isclose(p["value"], 100.0 - 30.0)
    assert math.isclose(p["variance"], 4.0 + 60.0**2 / 12.0)
    assert math.isclose(p["weight"], 1.0 / p["variance"])
    assert math.isclose(q["value"], 20.0 - 20.0)
    assert math.isclose(q["variance"], 4.0 + 20.0**2 / 12.0)
    assert p["status"] and q["status"]


def test_keeps_generation_only_injection_without_load_box():
    """Unset load box ([0, 0]) or no load on the node: measurement unchanged."""
    from gridstate.telemetry.load_only_injection import widen_generation_only_injections

    m = _load_node_model(load_p=(0.0, 0.0))
    stats = widen_generation_only_injections(m, {"P": frozenset({2}), "Q": frozenset({2})})
    assert stats == {"widened_p": 0, "widened_q": 0}
    assert _meas(m, 4)[0]["value"] == 100.0

    m = _load_node_model(load_p=(0.0, 60.0))
    nodes = m.nodes.to_numpy().copy()
    nodes["exist_load"][nodes["id"] == 2] = False
    m.nodes.update_from_array(nodes)
    assert widen_generation_only_injections(m, {"P": frozenset({2}), "Q": frozenset()}) == {
        "widened_p": 0,
        "widened_q": 0,
    }
