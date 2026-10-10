"""Warm start of ``gridstate.pipeline.run`` via an explicit ``init_state``.

Input is read-only => ``run`` clones the Input each time (V/delta flat), so a
cross-run warm start is given explicitly: the previous ``SEResult`` is passed as
``init_state`` and its V/delta are seeded into the working copy before preprocessing.

* unit: ``_seed_warm_start`` writes V/delta from the result into the clone;
* e2e synthetic: ``run(wls)`` -> ``run(ipm, init_state=res)`` without a crash + an event.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gridstate.pipeline import PipelineConfig, _build_working, _seed_warm_start, run


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
