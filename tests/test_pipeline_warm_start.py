"""Warm start of ``gridstate.pipeline.run`` via an explicit ``init_state``.

Input is read-only => ``run`` clones the Input each time (V/delta flat), so a
cross-run warm start is given explicitly: the previous ``SEResult`` is passed as
``init_state``.

* unit: ``_seed_warm_start`` writes V/delta from the result into the clone;
* e2e synthetic: ``run(wls)`` -> ``run(ipm, init_state=res)`` without a crash + an event;
* ``warm_start_mode``: ``"x0_only"`` (default) passes the previous solution as
  the explicit initial point (``estimate(x0=...)``) and the solve sees the
  cold-start problem; ``"seed_inputs"`` seeds V/delta into the node table
  before preprocessing, which also moves the pseudo-V priors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.pipeline import PipelineConfig, manifest, run
from gridstate.pipeline.runner import _build_working, _seed_warm_start
from gridstate.state import InitialPoint


# Reuse the synthetic model from the neighbouring module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_idempotent import _make_model_with_reactor


def test_seed_warm_start_writes_vd():
    """``_seed_warm_start`` carries V/delta from the previous result into the working copy."""
    m = _make_model_with_reactor()
    r = run(m, config=PipelineConfig(algorithm="wls"))

    # Replace V/delta in the previous result's Output table with recognisable values.
    out = r.outputs.nodes
    out["voltage_magnitude"] = 115.5
    out["voltage_angle"] = 0.037

    w = _build_working(m)
    seeded = _seed_warm_start(w, r)
    assert seeded == len(w.nodes.to_numpy())
    for n in w.nodes.to_numpy():
        assert float(n["voltage_magnitude"]) == pytest.approx(115.5)
        assert float(n["voltage_angle"]) == pytest.approx(0.037)


def test_run_with_init_state_no_crash_and_event():
    """``run(wls)`` -> ``run(ipm, init_state=res)``: no crash + a warm_start event."""
    m = _make_model_with_reactor()
    r_wls = run(m, config=PipelineConfig(algorithm="wls"))

    events: list[dict] = []
    r_ipm = run(
        m,
        config=PipelineConfig(algorithm="ipm"),
        init_state=r_wls,
        on_event=events.append,
    )
    assert r_ipm is not None
    warm = [e for e in events if e.get("type") == "warm_start"]
    assert warm and warm[0]["seeded_nodes"] == len(m.nodes.to_numpy())
    # The Input is still untouched.
    for n in m.nodes.to_numpy():
        assert float(n["voltage_magnitude"]) == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# warm_start_mode: configuration
# ---------------------------------------------------------------------------


def test_warm_start_mode_default_and_validation():
    assert PipelineConfig().warm_start_mode == "x0_only"
    PipelineConfig(warm_start_mode="seed_inputs").validate()
    with pytest.raises(ValueError, match="warm_start_mode"):
        PipelineConfig(warm_start_mode="x0")
    cfg = PipelineConfig()
    cfg.warm_start_mode = "both"
    with pytest.raises(ValueError, match="warm_start_mode"):
        run(_make_model_with_reactor(), config=cfg)


def test_warm_start_mode_in_manifest():
    params = {p["name"]: p for p in manifest()["params"]}
    entry = params["warm_start_mode"]
    assert entry["default"] == "x0_only"
    assert entry["choices"] == ["seed_inputs", "x0_only"]
    assert entry["control"] == "select"


# ---------------------------------------------------------------------------
# warm_start_mode: behaviour on IEEE inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def spy_estimate(monkeypatch):
    """Record, for every pipeline solve, the measurement/node tables it sees and its kwargs."""
    import gridstate.pipeline.passes as passes

    calls: list[dict] = []
    original = passes.estimate

    def _spy(model, **kwargs):
        calls.append(
            {
                "measurements": model.measurements.to_numpy().copy(),
                "nodes": model.nodes.to_numpy().copy(),
                "kwargs": dict(kwargs),
            }
        )
        return original(model, **kwargs)

    monkeypatch.setattr(passes, "estimate", _spy)
    return calls


def _ieee(name: str):
    from tests import _golden as golden

    return golden.load_ieee_input(name)


def _v_pu(result) -> np.ndarray:
    nodes = result.outputs.nodes
    order = np.argsort(nodes["id"])
    vn = {int(n["id"]): float(n["voltage_nominal"]) for n in result.model.nodes.to_numpy()}
    vn_arr = np.array([vn[int(i)] for i in nodes["id"][order]])
    return np.asarray(nodes["voltage_magnitude"][order] / vn_arr)


def _angles(result) -> np.ndarray:
    nodes = result.outputs.nodes
    return np.asarray(nodes["voltage_angle"][np.argsort(nodes["id"])])


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_x0_only_warm_start_solves_the_cold_problem(spy_estimate, algorithm):
    """x0_only: the main solve sees exactly the cold-start tables; only its start differs."""
    se = _ieee("case14_degraded")
    other = "ipm" if algorithm == "wls" else "wls"
    prev = run(se.model, config=PipelineConfig(algorithm=other))
    cfg = PipelineConfig(algorithm=algorithm, warm_start_mode="x0_only")

    spy_estimate.clear()
    cold = run(se.model, config=cfg)
    cold_call = spy_estimate[0]

    spy_estimate.clear()
    events: list[dict] = []
    warm = run(se.model, config=cfg, init_state=prev, on_event=events.append)
    warm_call = spy_estimate[0]

    # Same estimation problem: measurements incl. pseudo values/variances, node table.
    assert np.array_equal(warm_call["measurements"], cold_call["measurements"])
    assert np.array_equal(warm_call["nodes"], cold_call["nodes"])
    # The previous solution travels as the explicit initial point of the main solve only.
    x0 = warm_call["kwargs"]["x0"]
    assert isinstance(x0, InitialPoint)
    expected = InitialPoint.from_result(prev)
    assert np.array_equal(x0.voltage_magnitude, expected.voltage_magnitude)
    assert np.array_equal(x0.voltage_angle, expected.voltage_angle)
    assert warm_call["kwargs"]["init"] == "results"
    assert all("x0" not in c["kwargs"] for c in spy_estimate[1:])
    assert "x0" not in cold_call["kwargs"]
    seeded = [e["seeded_nodes"] for e in events if e["type"] == "warm_start"]
    assert seeded == [len(se.model.nodes.to_numpy())]

    # Same problem, different start: the end points agree to solver tolerance.
    assert warm.success and cold.success
    assert np.max(np.abs(_v_pu(warm) - _v_pu(cold))) < 1e-4
    assert np.max(np.abs(_angles(warm) - _angles(cold))) < 1e-4


def test_seed_inputs_warm_start_moves_the_pseudo_priors(spy_estimate):
    """seed_inputs (the previous default): the seeded V is the pseudo-V operating point."""
    se = _ieee("case14_degraded")
    cfg = PipelineConfig(warm_start_mode="seed_inputs")
    prev = run(se.model, config=cfg)

    spy_estimate.clear()
    run(se.model, config=cfg)
    cold = spy_estimate[0]["measurements"]
    spy_estimate.clear()
    run(se.model, config=cfg, init_state=prev)
    warm_call = spy_estimate[0]
    warm = warm_call["measurements"]

    assert "x0" not in warm_call["kwargs"]
    pseudo_cold = cold[cold["is_pseudo"].astype(bool)]
    pseudo_warm = warm[warm["is_pseudo"].astype(bool)]
    assert pseudo_cold.shape == pseudo_warm.shape
    assert not np.array_equal(pseudo_cold["value"], pseudo_warm["value"])


def test_x0_only_converges_to_the_cold_point_at_tight_tolerance():
    """WLS, tight tolerance: x0_only warm and cold end points coincide to roundoff."""
    se = _ieee("case30_degraded")
    cfg = PipelineConfig(tolerance=1e-8, warm_start_mode="x0_only")
    prev = run(se.model, config=PipelineConfig(algorithm="ipm"))
    cold = run(se.model, config=cfg)
    warm = run(se.model, config=cfg, init_state=prev)
    assert warm.success and cold.success
    assert np.max(np.abs(_v_pu(warm) - _v_pu(cold))) < 1e-10
    assert np.max(np.abs(_angles(warm) - _angles(cold))) < 1e-10


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_modes_agree_bit_for_bit_without_pseudo_v_priors(algorithm):
    """Fully measured input (no pseudo-V priors): both modes give the same start and result."""
    se = _ieee("case14")
    prev = run(se.model, config=PipelineConfig(algorithm="wls"))
    seed = run(
        se.model,
        config=PipelineConfig(algorithm=algorithm, warm_start_mode="seed_inputs"),
        init_state=prev,
    )
    x0 = run(se.model, config=PipelineConfig(algorithm=algorithm), init_state=prev)
    for name in ("voltage_magnitude", "voltage_angle", "p_inj_calc", "q_inj_calc"):
        assert np.array_equal(seed.outputs.nodes[name], x0.outputs.nodes[name]), name
    assert seed.iterations == x0.iterations


def test_init_state_accepts_an_initial_point():
    m = _make_model_with_reactor()
    prev = run(m, config=PipelineConfig(algorithm="wls"))
    a = run(m, config=PipelineConfig(algorithm="wls"), init_state=prev)
    b = run(m, config=PipelineConfig(algorithm="wls"), init_state=InitialPoint.from_result(prev))
    assert np.array_equal(
        a.outputs.nodes["voltage_magnitude"], b.outputs.nodes["voltage_magnitude"]
    )


# ---------------------------------------------------------------------------
# estimate(x0=...) and InitialPoint
# ---------------------------------------------------------------------------


def _working_before_solve():
    """A working model carrying every measurement of a cold run, V/δ at the input values."""
    se = _ieee("case14_degraded")
    prev = run(se.model, config=PipelineConfig(algorithm="ipm"))
    base = run(se.model, config=PipelineConfig())
    w = base.model.copy()
    nodes = w.nodes.to_numpy().copy()
    src = {int(n["id"]): n for n in se.model.nodes.to_numpy()}
    for i in range(len(nodes)):
        nodes["voltage_magnitude"][i] = src[int(nodes["id"][i])]["voltage_magnitude"]
        nodes["voltage_angle"][i] = src[int(nodes["id"][i])]["voltage_angle"]
    w.nodes.update_from_array(nodes)
    return w, InitialPoint.from_result(prev)


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_estimate_x0_equals_seeding_the_table(algorithm):
    """``estimate(x0=p)`` starts exactly where seeding ``p`` into the node table starts."""
    w, point = _working_before_solve()
    seeded = w.copy()
    _seed_warm_start(seeded, point)
    kw = {"algorithm": algorithm, "init": "results", "include_quality_summary": False}
    r_seed = estimate(seeded, **kw)
    r_x0 = estimate(w.copy(), x0=point, **kw)
    assert np.array_equal(r_seed.v_pu, r_x0.v_pu)
    assert np.array_equal(r_seed.delta_rad, r_x0.delta_rad)
    assert r_seed.iterations == r_x0.iterations


def test_estimate_x0_partial_cover_on_a_flat_base():
    """Nodes outside ``x0`` follow ``init``; a non-positive V starts at 1.0 p.u."""
    from gridstate.api import _build_initial_state
    from gridstate.state import StateLayout, unpack
    from gridstate.units import model_to_pu

    w, _point = _working_before_solve()
    net = model_to_pu(w)
    layout = StateLayout.from_slack(net.n_bus, net.slack_idx)
    first, second = int(net.bus_ids[0]), int(net.bus_ids[1])
    vn = {int(n["id"]): float(n["voltage_nominal"]) for n in w.nodes.to_numpy()}
    part = InitialPoint(
        node_ids=np.array([first, second]),
        voltage_magnitude=np.array([1.02 * vn[first], -1.0]),
        voltage_angle=np.array([-0.05, 0.07]),
    )
    delta, v = unpack(_build_initial_state(w, net, layout, "flat", x0=part), layout)
    assert v[0] == pytest.approx(1.02)
    assert v[1] == 1.0
    assert np.all(v[2:] == 1.0)
    for pos, angle in ((0, -0.05), (1, 0.07)):
        if pos != layout.slack_idx:
            assert delta[pos] == pytest.approx(angle)
    others = [i for i in range(net.n_bus) if i not in (0, 1)]
    assert np.all(delta[others] == 0.0)


def test_initial_point_validation_and_fallback():
    with pytest.raises(ValueError, match="same shape"):
        InitialPoint(np.array([1, 2]), np.array([1.0]), np.array([0.0, 0.0]))
    m = _make_model_with_reactor()
    r = run(m, config=PipelineConfig(algorithm="wls"))
    from_outputs = InitialPoint.from_result(r)
    r.outputs.nodes = r.outputs.nodes[:0]
    from_model = InitialPoint.from_result(r)
    assert np.array_equal(np.sort(from_outputs.node_ids), np.sort(from_model.node_ids))
    # A repeated id keeps its last entry.
    p = InitialPoint(np.array([7, 7]), np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    assert p.as_dict() == {7: (2.0, 0.2)}
