"""Plausibility check of the solved state (implausible V/Vnom) and its hints."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.pipeline import PipelineConfig, run
from gridstate.solution_hints import low_voltage_nodes, transit_injection_hints
from tests.test_boxed_balance import _build_net_with_transit


def test_low_voltage_nodes():
    v = np.array([1.0, 0.1, np.nan, 0.29, 0.31])
    assert low_voltage_nodes(v, 0.3).tolist() == [1, 3]
    assert low_voltage_nodes(v, 0.0).size == 0


@pytest.mark.parametrize("algorithm", ["wls", "ipm_pd"])
def test_implausible_state_is_not_a_success(caplog, algorithm):
    # A threshold above every solved V flags all nodes: the run is marked failed
    # and the warning names the nodes with a hint.
    m = _build_net_with_transit()
    with caplog.at_level(logging.WARNING, logger="gridstate.api"):
        res = estimate(
            m, algorithm=algorithm, include_quality_summary=False, implausible_v_ratio=2.0
        )
    assert not res.success
    assert res.message.startswith("implausible solution: V/Vnom < 2")
    text = "\n".join(r.getMessage() for r in caplog.records if "implausible" in r.getMessage())
    assert "node " in text and "p.u.;" in text


def test_check_disabled_keeps_the_solver_flag():
    m = _build_net_with_transit()
    res = estimate(m, algorithm="ipm_pd", include_quality_summary=False, implausible_v_ratio=0.0)
    assert res.success
    assert "implausible" not in res.message


def _nodes(p, q, exist_gen=0, exist_load=0, status=True):
    nodes = np.zeros(
        len(p),
        dtype=[
            ("id", "i8"),
            ("voltage_nominal", "f8"),
            ("status", "?"),
            ("exist_gen", "i1"),
            ("exist_load", "i1"),
            ("p_inj_calc", "f8"),
            ("q_inj_calc", "f8"),
        ],
    )
    nodes["id"] = np.arange(1, len(p) + 1)
    nodes["voltage_nominal"] = 220.0
    nodes["status"] = status
    nodes["exist_gen"] = exist_gen
    nodes["exist_load"] = exist_load
    nodes["p_inj_calc"] = p
    nodes["q_inj_calc"] = q
    return nodes


def test_transit_injection_hints_list_largest_first():
    nodes = _nodes([3.0, -0.5, 1.0, 0.0], [4.0, 0.2, -9.0, 0.0])
    lines = transit_injection_hints(nodes, 4.0)
    assert lines[0].startswith("2 transit node(s)")
    assert lines[1].startswith("  node 3 (220 kV): P = 1.0 MW, Q = -9.0 MVAr")
    assert lines[2].startswith("  node 1 ")
    assert len(lines) == 3


def test_transit_injection_hints_skip_marked_and_inactive_nodes():
    assert transit_injection_hints(_nodes([50.0], [0.0], exist_gen=1), 5.0) == []
    assert transit_injection_hints(_nodes([50.0], [0.0], exist_load=1), 5.0) == []
    assert transit_injection_hints(_nodes([50.0], [0.0], status=False), 5.0) == []
    assert transit_injection_hints(_nodes([50.0], [0.0]), 0.0) == []


def test_transit_injection_hints_cap_the_list():
    lines = transit_injection_hints(_nodes([10.0] * 4, [0.0] * 4), 5.0, top=2)
    assert lines[-1] == "  ... and 2 more"


def test_pipeline_warns_about_transit_injection(caplog):
    m = _build_net_with_transit()
    cfg = PipelineConfig(algorithm="ipm_pd", transit_injection_warn_mva=1e-9)
    with caplog.at_level(logging.WARNING, logger="gridstate.pipeline.runner"):
        run(m, config=cfg)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "transit node(s)" in text and "node 2 " in text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gridstate.pipeline.runner"):
        run(m, config=PipelineConfig(algorithm="ipm_pd", transit_injection_warn_mva=0.0))
    assert not any("transit node(s)" in r.getMessage() for r in caplog.records)
