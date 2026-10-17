"""Plausibility check of the solved state (implausible V/Vnom) and its hints."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.solution_hints import low_voltage_nodes
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
