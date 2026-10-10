"""Acceptance policy of post-solve re-solve passes (``gridstate.resolve_pass``).

A pass that edits measurements and warm re-solves must not turn a converged
solution into a non-converged one. The re-solve failure is forced by wrapping
``gridstate.pipeline.passes.estimate``: the first call (main solve) is untouched,
later calls (re-solves) run for real — so they do mutate V/δ, flows and
``estimated_si`` — and then report ``success=False``. After the revert the run
must be bit-identical to a run with that pass switched off.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import gridstate.pipeline.passes as passes_mod
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.pipeline import PipelineConfig
from gridstate.resolve_pass import (
    REVERT_REASON_NOT_CONVERGED,
    ResolvePass,
    ResolveSnapshot,
    resolve_regressed,
)
from gridstate.v_mirror import VMirrorPlan
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


_DATA = Path(__file__).parent / "test_data" / "ieee" / "case14.npz"

_PASSES_OFF: dict[str, Any] = {
    "bad_data": False,
    "v_refine": False,
    "v_mirror": False,
    "v_mirror_chain": False,
    "refine_merge": False,
    "anti_overshoot": False,
}


def _load(*, blind_every: int = 0) -> Any:
    """case14 input; ``blind_every=k`` drops real V on every k-th node."""
    se_in = load_se_input_npz(_DATA)
    if blind_every:
        model = se_in.model
        m = model.measurements.to_numpy()
        blind = model.nodes.to_numpy()["id"][::blind_every]
        sel = (
            (m["measurement_type"] == KIND_VOLTAGE)
            & (m["object_type"] == OBJ_NODE)
            & np.isin(m["object_id"], blind)
        )
        m["status"][sel] = False
        model.measurements.update_from_array(m)
    return se_in


def _fail_resolves(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make every estimate call after the first report non-convergence."""
    real = passes_mod.estimate
    calls: list[int] = []

    def _estimate(model: Any, **kw: Any) -> Any:
        res = real(model, **kw)
        calls.append(1)
        if len(calls) > 1:
            res.success = False
        return res

    monkeypatch.setattr(passes_mod, "estimate", _estimate)
    return calls


def _run(se_in: Any, cfg: PipelineConfig) -> tuple[Any, dict[str, dict]]:
    events: list[dict] = []
    out = run_se(se_in, config=cfg, on_event=events.append, validate=False)
    stats = {e["name"]: e["stats"] for e in events if e["type"] == "step_done"}
    return out, stats


def _assert_same_output(a: Any, b: Any) -> None:
    for name in ("nodes", "branches", "measurements"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        for col in x.dtype.names:
            assert np.array_equal(x[col], y[col], equal_nan=x[col].dtype.kind == "f"), (name, col)
    assert np.array_equal(a.v_pu, b.v_pu)
    assert np.array_equal(a.delta_rad, b.delta_rad)
    assert (a.success, a.iterations, a.objective_value) == (
        b.success,
        b.iterations,
        b.objective_value,
    )
    for table in ("nodes", "branches", "measurements"):
        x = getattr(a.result.model, table).to_numpy()
        y = getattr(b.result.model, table).to_numpy()
        assert x.dtype == y.dtype and x.shape == y.shape, table
        for col in x.dtype.names:
            assert np.array_equal(x[col], y[col], equal_nan=x[col].dtype.kind == "f"), (table, col)


def _check_revert(
    monkeypatch: pytest.MonkeyPatch,
    *,
    se_in_factory: Any,
    cfg: PipelineConfig,
    reference_cfg: PipelineConfig,
    step: str,
) -> dict:
    ref, _ = _run(se_in_factory(), reference_cfg)
    assert ref.success
    calls = _fail_resolves(monkeypatch)
    out, stats = _run(se_in_factory(), cfg)
    assert len(calls) >= 2, "the pass under test did not re-solve"
    st = stats[step]
    assert st["success"] is False
    assert st["reverted"] is True
    assert st["revert_reason"] == REVERT_REASON_NOT_CONVERGED
    assert out.success is True
    _assert_same_output(out, ref)
    return st


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_bad_data_repass_reverts_on_failed_resolve(monkeypatch, algorithm):
    base = replace(PipelineConfig(algorithm=algorithm), **_PASSES_OFF)
    st = _check_revert(
        monkeypatch,
        se_in_factory=_load,
        cfg=replace(base, bad_data=True, bad_data_threshold=2.0),
        reference_cfg=base,
        step="bad_data_repass",
    )
    assert st["flips"] + st["rejects"] + st["damped"] > 0


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_v_refine_reverts_on_failed_resolve(monkeypatch, algorithm):
    base = replace(PipelineConfig(algorithm=algorithm), **_PASSES_OFF)
    st = _check_revert(
        monkeypatch,
        se_in_factory=_load,
        cfg=replace(base, v_refine=True),
        reference_cfg=base,
        step="v_refine",
    )
    assert st["tightened"] > 0


def _pseudo_v_plan(measurements: np.ndarray, *_a: Any, **_kw: Any) -> VMirrorPlan:
    """Non-empty mirror plan: lift every active pseudo-V by 2 %."""
    m = measurements
    sel = (
        m["status"].astype(bool)
        & m["is_pseudo"].astype(bool)
        & (m["measurement_type"] == KIND_VOLTAGE)
        & (m["object_type"] == OBJ_NODE)
    )
    vals = tuple((int(m["object_id"][j]), float(m["value"][j]) * 1.02) for j in np.where(sel)[0])
    return VMirrorPlan(vals, n_clusters=1)


def test_v_mirror_reverts_on_failed_resolve(monkeypatch):
    base = replace(PipelineConfig(algorithm="wls"), **_PASSES_OFF)
    # The IEEE fixture has no blind clusters with a boundary; force a plan.
    monkeypatch.setattr(passes_mod, "classify_v_mirror", _pseudo_v_plan)
    st = _check_revert(
        monkeypatch,
        se_in_factory=lambda: _load(blind_every=2),
        cfg=replace(base, v_mirror=True),
        reference_cfg=base,
        step="v_mirror",
    )
    assert st["nodes"] > 0


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_v_mirror_chain_reverts_on_failed_resolve(monkeypatch, algorithm):
    base = replace(PipelineConfig(algorithm=algorithm), **_PASSES_OFF)
    st = _check_revert(
        monkeypatch,
        se_in_factory=lambda: _load(blind_every=2),
        cfg=replace(base, v_mirror_chain=True),
        reference_cfg=base,
        step="v_mirror_chain",
    )
    assert st["nodes"] > 0


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_refine_merged_reverts_all_plans_on_failed_resolve(monkeypatch, algorithm):
    base = replace(PipelineConfig(algorithm=algorithm), **_PASSES_OFF)
    st = _check_revert(
        monkeypatch,
        se_in_factory=lambda: _load(blind_every=2),
        cfg=replace(base, refine_merge=True, v_refine=True, v_mirror=True, v_mirror_chain=True),
        reference_cfg=base,
        step="refine_merged",
    )
    # two plans applied before the single re-solve; both are reverted
    assert {"v_refine", "v_mirror_chain"} <= st.keys()


def test_anti_overshoot_rejects_failed_resolve_in_pipeline(monkeypatch):
    """Ceiling 1.0 makes anti-overshoot tighten blind nodes; a failed re-solve
    must end in a full revert to the converged base solution."""
    base = replace(PipelineConfig(algorithm="wls"), **_PASSES_OFF)
    st = _check_revert_anti_overshoot(monkeypatch, base)
    assert st["tightened"] > 0
    assert st["accepted"] is False
    assert st["revert_reason"] == REVERT_REASON_NOT_CONVERGED


def _check_revert_anti_overshoot(monkeypatch: pytest.MonkeyPatch, base: PipelineConfig) -> dict:
    ref, _ = _run(_load(blind_every=2), base)
    assert ref.success
    _fail_resolves(monkeypatch)
    out, stats = _run(
        _load(blind_every=2), replace(base, anti_overshoot=True, anti_overshoot_ceiling=1.0)
    )
    assert out.success is True
    _assert_same_output(out, ref)
    return stats["anti_overshoot"]


def test_successful_resolve_carries_no_revert_keys():
    cfg = replace(PipelineConfig(algorithm="wls"), **_PASSES_OFF)
    _, stats = _run(_load(), replace(cfg, v_refine=True, bad_data=True, bad_data_threshold=2.0))
    for step in ("bad_data_repass", "v_refine"):
        assert stats[step]["success"] is True
        assert "reverted" not in stats[step]
        assert "revert_reason" not in stats[step]


# --- helper unit tests ------------------------------------------------------


def test_snapshot_does_not_alias_live_tables():
    model = _load().model
    snap = ResolveSnapshot.capture(model)
    v0 = snap.nodes["voltage_magnitude"].copy()
    var0 = snap.measurements["variance"].copy()

    # in-place edit through a row proxy and a full write-back
    nid = int(model.nodes.to_numpy()["id"][0])
    model.nodes.get_by_id(nid).voltage_magnitude = -1.0
    m = model.measurements.to_numpy()
    m["variance"] *= 4.0
    model.measurements.update_from_array(m)
    assert np.array_equal(snap.nodes["voltage_magnitude"], v0)
    assert np.array_equal(snap.measurements["variance"], var0)

    snap.restore(model)
    assert np.array_equal(model.nodes.to_numpy()["voltage_magnitude"], v0)
    # editing the restored model must not reach the snapshot either
    model.nodes.get_by_id(nid).voltage_magnitude = -2.0
    assert np.array_equal(snap.nodes["voltage_magnitude"], v0)
    snap.restore(model)
    assert np.array_equal(model.nodes.to_numpy()["voltage_magnitude"], v0)


@pytest.mark.parametrize(
    ("before", "after", "regressed"),
    [(True, True, False), (True, False, True), (False, False, False), (False, True, False)],
)
def test_resolve_regressed_rule(before, after, regressed):
    b = SimpleNamespace(success=before)
    a = SimpleNamespace(success=after)
    assert resolve_regressed(b, a) is regressed  # type: ignore[arg-type]


def test_guard_keeps_failed_resolve_when_base_failed_too():
    """No usable state to fall back on → the re-solve stays current."""
    model = _load().model
    base = SimpleNamespace(success=False)
    resolved = SimpleNamespace(success=False)
    guard = ResolvePass(model, base)  # type: ignore[arg-type]
    guard.arm()
    stats: dict = {}
    assert guard.settle(resolved, stats) is resolved  # type: ignore[arg-type]
    assert stats == {}


def test_guard_arm_is_idempotent_and_required():
    model = _load().model
    v0 = model.nodes.to_numpy()["voltage_magnitude"].copy()
    guard = ResolvePass(model, SimpleNamespace(success=True))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        guard.settle(SimpleNamespace(success=False), {})  # type: ignore[arg-type]
    guard.arm()
    nid = int(model.nodes.to_numpy()["id"][0])
    model.nodes.get_by_id(nid).voltage_magnitude = -1.0
    guard.arm()  # second arm must not re-capture the edited state
    stats: dict = {}
    assert guard.settle(SimpleNamespace(success=False), stats) is guard.base  # type: ignore[arg-type]
    assert stats["reverted"] is True
    assert np.array_equal(model.nodes.to_numpy()["voltage_magnitude"], v0)


def test_resolve_pass_guard_and_revert():
    model = _load().model
    assert ResolvePass(model, SimpleNamespace(success=True)).usable  # type: ignore[arg-type]
    assert not ResolvePass(model, SimpleNamespace(success=False)).usable  # type: ignore[arg-type]
    rp = ResolvePass(model, SimpleNamespace(success=True))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        rp.revert()
    v0 = model.nodes.to_numpy()["voltage_magnitude"].copy()
    rp.arm()
    nid = int(model.nodes.to_numpy()["id"][0])
    model.nodes.get_by_id(nid).voltage_magnitude = -1.0
    assert rp.revert() is rp.base
    assert np.array_equal(model.nodes.to_numpy()["voltage_magnitude"], v0)


def test_both_refine_paths_share_the_stage_objects():
    """The merged step runs the very stages of the sequential steps, gated by
    the toggles of those steps."""
    stages = passes_mod.MERGED_STAGES
    assert stages == (
        passes_mod.V_REFINE_STAGE,
        passes_mod.V_MIRROR_STAGE,
        passes_mod.V_MIRROR_CHAIN_STAGE,
    )
    cfg = PipelineConfig()
    assert all(isinstance(getattr(cfg, s.name), bool) for s in stages)
