"""Range check of a solved node state (``gridstate.resolve_pass.state_range_violation``).

A solve can report success and still leave a state the p.u. conversion
rejects (angles beyond ±2π) or that is physically meaningless (non-finite or
non-positive V). Such a state must never become the current solution of a
re-solve pass, and when the main estimate itself returns one, the pipeline
must end with ``success=False`` instead of raising in the quality summary.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import gridstate.pipeline.passes as passes_mod
from gridstate.contract import run as run_se
from gridstate.pipeline import PipelineConfig
from gridstate.post_processing import refine_anti_overshoot
from gridstate.resolve_pass import (
    MIN_VOLTAGE_RATIO,
    REVERT_REASON_NOT_CONVERGED,
    REVERT_REASON_OUT_OF_RANGE,
    ResolvePass,
    state_range_violation,
)
from gridstate.units import MAX_ABS_VOLTAGE_ANGLE, network_pu_from_tables
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
)
from tests import _golden as golden
from tests.test_anti_overshoot import _build_model, _count_pseudo_inj


_PASSES_OFF: dict[str, Any] = {
    "bad_data": False,
    "v_refine": False,
    "v_mirror": False,
    "v_mirror_chain": False,
    "refine_merge": False,
    "anti_overshoot": False,
}


def _gross_errors(name: str, seed: int) -> Any:
    """IEEE input with gross errors (+-500..2000 MW/MVAr) on 10 % of the P/Q rows."""
    se_in = golden.load_ieee_input(name)
    arr = se_in.model.measurements.to_numpy()
    rng = np.random.default_rng(seed)
    kinds = [KIND_POWER_P, KIND_POWER_Q, KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q]
    pq = np.flatnonzero(np.isin(arr["measurement_type"], kinds) & arr["status"])
    k = rng.choice(pq, size=max(1, len(arr) // 10), replace=False)
    arr["value"][k] += rng.choice([-1, 1], size=k.size) * rng.uniform(500.0, 2000.0, size=k.size)
    se_in.model.measurements.update_from_array(arr)
    return se_in


def _run(se_in: Any, cfg: PipelineConfig) -> tuple[Any, dict[str, dict]]:
    events: list[dict] = []
    out = run_se(se_in, config=cfg, on_event=events.append, validate=False)
    stats = {e["name"]: e["stats"] for e in events if e["type"] == "step_done"}
    return out, stats


def _nodes() -> np.ndarray:
    return golden.load_ieee_input("case14").model.nodes.to_numpy()


def _active_angles(model: Any) -> np.ndarray:
    nd = model.nodes.to_numpy()
    return nd["voltage_angle"][nd["status"]]


# --- the predicate ------------------------------------------------------------


def test_clean_state_is_in_range():
    assert state_range_violation(_nodes()) is None


@pytest.mark.parametrize(
    ("column", "value", "what"),
    [
        ("voltage_magnitude", np.nan, "non-finite"),
        ("voltage_magnitude", np.inf, "non-finite"),
        ("voltage_angle", np.nan, "non-finite"),
        ("voltage_magnitude", 0.0, "V/Vnom"),
        ("voltage_magnitude", -1.0, "V/Vnom"),
        ("voltage_angle", MAX_ABS_VOLTAGE_ANGLE + 1e-9, "angle"),
        ("voltage_angle", -MAX_ABS_VOLTAGE_ANGLE - 1e-9, "angle"),
    ],
)
def test_out_of_range_values_are_reported(column, value, what):
    nd = _nodes()
    nd[column][3] = value
    msg = state_range_violation(nd)
    assert msg is not None
    assert what in msg
    assert f"[{int(nd['id'][3])}]" in msg


def test_bounds_are_inclusive():
    nd = _nodes()
    nd["voltage_angle"][1] = MAX_ABS_VOLTAGE_ANGLE
    nd["voltage_angle"][2] = -MAX_ABS_VOLTAGE_ANGLE
    nd["voltage_magnitude"][3] = MIN_VOLTAGE_RATIO * nd["voltage_nominal"][3]
    assert state_range_violation(nd) is None
    nd["voltage_magnitude"][3] *= 0.999
    assert "V/Vnom" in (state_range_violation(nd) or "")


def test_angle_bound_is_the_one_of_the_pu_conversion():
    """In range for the predicate <=> accepted by the p.u. conversion."""
    assert math.isclose(MAX_ABS_VOLTAGE_ANGLE, 2 * math.pi)
    se_in = golden.load_ieee_input("case14")
    branches = se_in.model.branches.to_numpy()
    nd = se_in.model.nodes.to_numpy()
    nd["voltage_angle"][4] = MAX_ABS_VOLTAGE_ANGLE
    network_pu_from_tables(nd, branches)
    assert state_range_violation(nd) is None
    nd["voltage_angle"][4] = np.nextafter(MAX_ABS_VOLTAGE_ANGLE, np.inf)
    with pytest.raises(ValueError, match="voltage_angle"):
        network_pu_from_tables(nd, branches)
    assert state_range_violation(nd) is not None


def test_inactive_nodes_and_missing_vnom_are_ignored():
    nd = _nodes()
    nd["status"][5] = False
    nd["voltage_magnitude"][5] = np.nan
    nd["voltage_angle"][5] = 100.0
    nd["voltage_nominal"][6] = 0.0  # left to the p.u. conversion
    nd["voltage_magnitude"][6] = 0.0
    assert state_range_violation(nd) is None


def test_several_problems_and_quoted_ids_are_capped():
    nd = _nodes()
    nd["voltage_angle"][:7] = 10.0
    nd["voltage_magnitude"][8] = np.nan
    msg = state_range_violation(nd) or ""
    assert "non-finite V/angle at 1 node(s)" in msg
    assert f"|angle| > 2*pi at 7 node(s) {nd['id'][:5].tolist()}, ..." in msg


# --- the shared acceptance ----------------------------------------------------


def test_settle_rejects_an_out_of_range_resolve_that_converged():
    model = golden.load_ieee_input("case14").model
    nodes0 = model.nodes.to_numpy()
    rp = ResolvePass(model, SimpleNamespace(success=True))  # type: ignore[arg-type]
    rp.arm()
    nid = int(nodes0["id"][3])
    model.nodes.get_by_id(nid).voltage_angle = 7.0
    stats: dict = {}
    assert rp.settle(SimpleNamespace(success=True), stats) is rp.base  # type: ignore[arg-type]
    assert stats["reverted"] is True
    assert stats["revert_reason"] == REVERT_REASON_OUT_OF_RANGE
    assert f"[{nid}]" in stats["state_violation"]
    assert (model.nodes.to_numpy() == nodes0).all()


def test_settle_rejects_an_out_of_range_resolve_even_from_a_failed_base():
    """Unlike non-convergence, an out-of-range state is never kept: no later
    step can start from it, so the snapshot is never worse."""
    model = golden.load_ieee_input("case14").model
    rp = ResolvePass(model, SimpleNamespace(success=False))  # type: ignore[arg-type]
    rp.arm()
    model.nodes.get_by_id(int(model.nodes.to_numpy()["id"][2])).voltage_magnitude = 0.0
    stats: dict = {}
    assert rp.settle(SimpleNamespace(success=True), stats) is rp.base  # type: ignore[arg-type]
    assert stats["revert_reason"] == REVERT_REASON_OUT_OF_RANGE


def test_pipeline_pass_reverts_out_of_range_resolve(monkeypatch):
    """A converged re-solve of a pipeline pass that lands out of range is
    reverted; the run equals the run with the pass off."""
    base = replace(PipelineConfig(algorithm="wls"), **_PASSES_OFF)
    ref, _ = _run(golden.load_ieee_input("case14"), base)
    real = passes_mod.estimate
    calls: list[int] = []

    def _estimate(model: Any, **kw: Any) -> Any:
        res = real(model, **kw)
        calls.append(1)
        if len(calls) > 1:  # re-solves: shift one angle beyond 2*pi
            nid = int(model.nodes.to_numpy()["id"][3])
            model.nodes.get_by_id(nid).voltage_angle = 7.0
        return res

    monkeypatch.setattr(passes_mod, "estimate", _estimate)
    out, stats = _run(
        golden.load_ieee_input("case14"), replace(base, bad_data=True, bad_data_threshold=2.0)
    )
    assert len(calls) == 2
    st = stats["bad_data_repass"]
    assert st["success"] is True  # the solver's verdict is kept in the stats
    assert st["reverted"] is True
    assert st["revert_reason"] == REVERT_REASON_OUT_OF_RANGE
    assert out.success is True
    assert np.array_equal(out.v_pu, ref.v_pu)
    assert np.array_equal(out.delta_rad, ref.delta_rad)
    assert np.array_equal(out.result.model.nodes.to_numpy(), ref.result.model.nodes.to_numpy())


def test_anti_overshoot_rejects_out_of_range_resolve_and_stops():
    m = _build_model(overshoot_pu=1.30)
    nodes0 = m.nodes.to_numpy()
    calls = {"n": 0}

    def resolve():
        calls["n"] += 1
        # lowers max V/Vnom (would be accepted) but leaves an angle beyond 2*pi
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.05, "voltage_angle": 7.0})
        return SimpleNamespace(success=True)

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"
    assert calls["n"] == 1  # no further round from an out-of-range state
    assert stats["accepted"] is False
    assert stats["revert_reason"] == REVERT_REASON_OUT_OF_RANGE
    assert "angle" in stats["state_violation"]
    assert stats["max_ratio_after"] < stats["max_ratio_before"]
    assert (m.nodes.to_numpy() == nodes0).all()
    assert _count_pseudo_inj(m) == 0


def test_anti_overshoot_non_converged_in_range_keeps_its_reason():
    m = _build_model(overshoot_pu=1.30)

    def resolve():
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.05})
        return SimpleNamespace(success=False)

    _, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)
    assert stats["revert_reason"] == REVERT_REASON_NOT_CONVERGED
    assert "state_violation" not in stats


# --- regressions on gross errors (library defaults) ---------------------------


def test_defaults_gross_errors_anti_overshoot_out_of_range_is_rejected():
    """case14_degraded + gross errors, seed 92: the anti-overshoot re-solve
    ends out of range (a collapsed V); such a state used to be accepted and the
    quality summary then raised ValueError in the p.u. conversion."""
    out, stats = _run(_gross_errors("case14_degraded", seed=92), PipelineConfig())
    st = stats["anti_overshoot"]
    assert st["tightened"] > 0
    assert st["accepted"] is False
    assert st["revert_reason"] == REVERT_REASON_OUT_OF_RANGE
    assert st["state_violation"]
    # the base solution stays current and is usable
    assert out.success is stats["estimate"]["success"]
    assert stats["quality_summary"] == {}
    assert out.result.chi2 is not None
    assert state_range_violation(out.result.model.nodes.to_numpy()) is None
    assert np.all(np.abs(_active_angles(out.result.model)) <= MAX_ABS_VOLTAGE_ANGLE)


def test_defaults_gross_errors_estimate_out_of_range_fails_cleanly():
    """case14 + gross errors, seed 10: the main WLS estimate reports success
    with a negative V; the run fails with a message instead of going on."""
    out, stats = _run(_gross_errors("case14", seed=10), PipelineConfig())
    assert out.success is False
    assert "out of range" in out.result.message
    assert stats["estimate"]["success"] is False
    assert "V/Vnom" in stats["estimate"]["state_violation"]
    assert "skipped" in stats["anti_overshoot"]
    assert "skipped" in stats["quality_summary"]


def test_estimate_angle_out_of_range_does_not_raise(monkeypatch):
    """Main estimate leaving an angle beyond 2*pi: no re-solve starts from it
    and the quality summary is skipped instead of raising."""
    real = passes_mod.estimate

    def _estimate(model: Any, **kw: Any) -> Any:
        res = real(model, **kw)
        nid = int(model.nodes.to_numpy()["id"][3])
        model.nodes.get_by_id(nid).voltage_angle = 7.0
        return res

    monkeypatch.setattr(passes_mod, "estimate", _estimate)
    cfg = replace(PipelineConfig(), bad_data=True, anti_overshoot=True)
    out, stats = _run(golden.load_ieee_input("case14"), cfg)
    assert out.success is False
    assert "angle" in stats["estimate"]["state_violation"]
    assert "skipped" in stats["bad_data_repass"]
    assert "skipped" in stats["anti_overshoot"]
    assert "angle" in stats["quality_summary"]["skipped"]
