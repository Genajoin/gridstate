"""Unit tests for ``gridstate.post_processing.refine_anti_overshoot``.

Check the self-validating anti-overshoot refinement logic WITHOUT a full solve:
the ``resolve`` callback is replaced by a fake that mutates ``voltage_magnitude``
(imitating a re-solve). This tests three branches deterministically:
  * accept -- refine lowered max(V/Vnom) -> accepted (refined result, V lowered, tight
    P/Q injection pseudo-measurements added);
  * revert -- refine did NOT lower max(V/Vnom) -> rollback (base result, V restored);
  * no-op -- no node > ceiling -> nothing added, base result.
"""

from __future__ import annotations

from types import SimpleNamespace

from gridstate.constants import (
    MeasurementObjectType,
    MeasurementType,
    NodeType,
)
from gridstate.post_processing import _max_voltage_ratio, refine_anti_overshoot
from gridstate.resolve_pass import REVERT_REASON_NOT_CONVERGED
from gridstate.working import Working


# Stand-in for a converged re-solve result (only ``success`` is read).
_REFINED = SimpleNamespace(success=True)


def _build_model(*, overshoot_pu: float = 1.30):
    """2 nodes: slack (real V, 1.0 pu) + a PQ node with V overshoot and NO real V measurement."""
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0 * overshoot_pu,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
            "load_p": 5.0,
            "load_q": 2.0,
            "generation_p": 0.0,
            "generation_q": 0.0,
        }
    )
    # real V measurement ONLY on the slack -- node 2 is "V-unobservable" (a lever candidate).
    m.measurements.add(
        {
            "id": 1,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": 1,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": 110.0,
            "variance": 0.1,
            "is_pseudo": False,
            "status": True,
        }
    )
    return m


def _count_pseudo_inj(model) -> int:
    me = model.measurements.to_numpy()
    return int(
        sum(
            1
            for i in range(len(me))
            if bool(me["is_pseudo"][i])
            and int(me["measurement_type"][i])
            in (int(MeasurementType.POWER_INJECTION_P), int(MeasurementType.POWER_INJECTION_Q))
        )
    )


def test_anti_overshoot_accept_lowers_v():
    """resolve lowers V of node 2 -> max V drops -> refine accepted."""
    m = _build_model(overshoot_pu=1.30)

    def resolve():
        # imitate the effect of a tight injection: V settles to the physical level 1.05 pu
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.05})
        return _REFINED

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result is _REFINED
    assert stats["accepted"] is True
    assert stats["tightened"] == 1
    assert stats["max_ratio_before"] > stats["max_ratio_after"]
    # node 2 stays lowered
    assert abs(_max_voltage_ratio(m) - 1.05) < 1e-6
    # P_inj and Q_inj pseudo-measurements added on the overshoot node
    assert _count_pseudo_inj(m) == 2


def test_anti_overshoot_revert_when_not_improved():
    """resolve does NOT lower V (even raises it) -> max V did not drop -> rollback to base."""
    m = _build_model(overshoot_pu=1.30)

    def resolve():
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.35})  # got worse
        return _REFINED

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"  # rollback
    assert stats["accepted"] is False
    # V of node 2 restored to the original overshoot 1.30 (snapshot rollback)
    assert abs(_max_voltage_ratio(m) - 1.30) < 1e-6


def test_anti_overshoot_stops_after_round_that_did_not_help():
    """The pass did not lower max V below the base but raised a new node above the ceiling --
    no second pass is made, the outcome is a rollback."""
    m = _build_model(overshoot_pu=1.30)
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    calls = []

    def resolve():
        calls.append(1)
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.10})
        m.nodes.update(3, {"voltage_magnitude": 110.0 * 1.40})  # new overshoot
        return _REFINED

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert len(calls) == 1
    assert result == "BASE"
    assert stats["accepted"] is False
    assert stats["tightened"] == 1


def test_anti_overshoot_noop_when_no_overshoot():
    """No node > ceiling -> resolve is not called, nothing is added."""
    m = _build_model(overshoot_pu=1.05)  # within the norm
    calls = {"n": 0}

    def resolve():
        calls["n"] += 1
        return _REFINED

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"
    assert stats["tightened"] == 0
    assert stats["accepted"] is False
    assert calls["n"] == 0  # resolve was not called
    assert _count_pseudo_inj(m) == 0


def test_anti_overshoot_skips_node_with_real_v():
    """A node with a real V measurement is NOT touched even on overshoot (observed -- not an artefact)."""
    m = _build_model(overshoot_pu=1.30)
    # give node 2 a real V measurement -> excluded from candidates
    m.measurements.add(
        {
            "id": 2,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": 2,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": 110.0 * 1.30,
            "variance": 0.1,
            "is_pseudo": False,
            "status": True,
        }
    )
    calls = {"n": 0}

    def resolve():
        calls["n"] += 1
        return _REFINED

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"
    assert stats["tightened"] == 0
    assert calls["n"] == 0


def test_anti_overshoot_rejects_non_converged_resolve():
    """A re-solve that lowers max V/Vnom but did not converge is NOT accepted:
    full revert (V, added priors) to the base result."""
    m = _build_model(overshoot_pu=1.30)
    nodes0 = m.nodes.to_numpy()
    meas0 = m.measurements.to_numpy()
    failed = SimpleNamespace(success=False)

    def resolve():
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.05})
        return failed

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"
    assert stats["accepted"] is False
    assert stats["tightened"] == 1
    assert stats["revert_reason"] == REVERT_REASON_NOT_CONVERGED
    # the ratio did drop — the rejection is due to non-convergence alone
    assert stats["max_ratio_after"] < stats["max_ratio_before"]
    assert (m.nodes.to_numpy() == nodes0).all()
    assert len(m.measurements) == len(meas0)
    assert _count_pseudo_inj(m) == 0


def test_anti_overshoot_accepts_converged_resolve_without_revert_reason():
    m = _build_model(overshoot_pu=1.30)

    def resolve():
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.05})
        return _REFINED

    _, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert stats["accepted"] is True
    assert "revert_reason" not in stats
