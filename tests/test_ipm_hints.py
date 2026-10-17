"""Convergence hints of slow / non-converged IPM solves."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import gridstate.api as api
from gridstate.algorithms.ipm import IPMResult
from gridstate.api import estimate
from gridstate.ipm_hints import _variable, has_dominant_limiter
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.state import StateLayout
from gridstate.units import model_to_pu
from gridstate.z_vector import KIND_BOX_PRIOR_PGEN, build_z_and_r
from tests.test_boxed_balance import _build_net_with_transit


def _with_limiter(limiter, **solver_overrides):
    """Wrap ``solve_ipm_pd``: apply ``solver_overrides`` and inject a step limiter."""
    real = api.solve_ipm_pd

    def wrapped(*args, **kwargs):
        out = real(*args, **{**kwargs, **solver_overrides})
        out.step_limiters = {limiter(kwargs): out.iterations_inner_total}
        return out

    return wrapped


def _first_box(kwargs) -> int:
    return int(kwargs["box_idx"][0])


def _hints(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "convergence hints" in r.getMessage()]


def test_fallback_logs_hints_naming_the_limiting_node(monkeypatch, caplog):
    monkeypatch.setattr(api, "solve_ipm_pd", _with_limiter(_first_box, max_iter=1))
    m = _build_net_with_transit()
    with caplog.at_level(logging.WARNING, logger="gridstate.api"):
        res = estimate(m, algorithm="ipm_pd", include_quality_summary=False)
    assert res.success
    (text,) = _hints(caplog)
    assert "ipm_pd fell back to the primal IPM" in text
    assert "step limited most often by:" in text
    assert "1 of 1 iterations" in text


def test_fast_solve_logs_no_hints(caplog):
    m = _build_net_with_transit()
    with caplog.at_level(logging.WARNING, logger="gridstate.api"):
        res = estimate(m, algorithm="ipm_pd", include_quality_summary=False)
    assert res.success
    assert _hints(caplog) == []


@pytest.mark.parametrize(("share", "expected"), [(1.0, 1), (0.0, 0)])
def test_iteration_trigger_needs_a_dominant_limiter(monkeypatch, caplog, share, expected):
    real = api.solve_ipm_pd

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        n = out.iterations_inner_total
        out.step_limiters = {_first_box(kwargs): int(share * n)} if share else {}
        return out

    monkeypatch.setattr(api, "solve_ipm_pd", wrapped)
    m = _build_net_with_transit()
    with caplog.at_level(logging.WARNING, logger="gridstate.api"):
        estimate(m, algorithm="ipm_pd", include_quality_summary=False, hint_iterations=1)
    texts = _hints(caplog)
    assert len(texts) == expected
    if expected:
        assert "iterations > 1" in texts[0]


def test_has_dominant_limiter():
    res = IPMResult(
        x=np.zeros(1),
        success=True,
        objective_data=0.0,
        iterations_outer=1,
        iterations_inner_total=1,
        mu_final=0.0,
        grad_inf_final=0.0,
    )
    assert not has_dominant_limiter(res, 100)
    res.step_limiters = {3: 9}
    assert not has_dominant_limiter(res, 100)
    res.limiting_rows = {7: 10}
    assert has_dominant_limiter(res, 100)


def test_state_index_maps_to_node_and_variable():
    m = _build_net_with_transit()
    net = model_to_pu(m)
    layout = StateLayout.from_slack(net.n_bus, net.slack_idx)
    z, r, mi = build_z_and_r(m, m.measurements, net)
    setup = build_ipm_setup(m, net, z, r, mi, layout_base=layout)
    lay = setup.layout
    assert _variable(setup, lay.offset_delta) == (int(lay.non_slack_idx[0]), -1)
    assert _variable(setup, lay.offset_v + 1) == (1, -2)
    if lay.pgen_node_pos.size:
        assert _variable(setup, lay.offset_pgen) == (
            int(lay.pgen_node_pos[0]),
            KIND_BOX_PRIOR_PGEN,
        )
    with pytest.raises(IndexError):
        _variable(setup, lay.size)
