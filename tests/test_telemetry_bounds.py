"""Plausible ranges of telemetry values (``DerivedInputs.telemetry_bounds``) in the kernel."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.bounds import SENTINEL_VALUE
from gridstate.constants import SOURCE_INJ_GEN, SOURCE_INJ_LOAD, MeasurementQuality
from tests.test_telemetry_apply_kernel import _branches, _meas, _nodes, _run


GOOD = MeasurementQuality.GOOD


def _flow_case():
    nodes = _nodes([{"id": 1, "voltage_nominal": 220.0}])
    branches = _branches([{"id": 7, "from_node": 1, "status": True}])
    meas = _meas(
        [
            {"id": 10, "object_type": 0, "object_id": 1, "measurement_type": 2},
            {"id": 11, "object_type": 1, "object_id": 7, "measurement_type": 0, "branch_side": 0},
        ]
    )
    resolved = {(1, "v"): (225.0, 1, "g-v", GOOD), (7, "p_from"): (150.0, 1, "g-p", GOOD)}
    return meas, nodes, branches, resolved


def test_none_leaves_the_rows_untouched():
    meas, nodes, branches, resolved = _flow_case()
    a, b = meas.copy(), meas.copy()
    _, rows_a = _run(a, nodes, branches, resolved)
    _, rows_b = _run(b, nodes, branches, resolved, telemetry_bounds={})
    np.testing.assert_array_equal(a, b)
    assert rows_a == rows_b


def test_voltage_and_flow_rows_get_their_range():
    meas, nodes, branches, resolved = _flow_case()
    bounds = {(1, "v"): (200.0, 250.0), (7, "p_from"): (-np.inf, 300.0)}
    _run(meas, nodes, branches, resolved, telemetry_bounds=bounds)
    assert (meas["min_value"][0], meas["max_value"][0]) == (200.0, 250.0)
    assert (meas["min_value"][1], meas["max_value"][1]) == (-SENTINEL_VALUE, 300.0)


def test_injection_rows_combine_the_component_ranges():
    nodes = _nodes([{"id": 5, "voltage_nominal": 220.0}])
    meas = _meas([{"id": 10, "object_type": 0, "object_id": 5, "measurement_type": 2}])
    resolved = {
        (5, "p_gen:0"): (200.0, 1, "g0", GOOD),
        (5, "p_gen:1"): (100.0, 1, "g1", GOOD),
        (5, "p_load"): (120.0, 1, "pn", GOOD),
        (5, "q_load"): (30.0, 1, "qn", GOOD),
    }
    bounds = {
        (5, "p_gen:0"): (0.0, 250.0),
        (5, "p_gen:1"): (10.0, 150.0),
        (5, "p_load"): (50.0, 200.0),
    }
    _, rows = _run(
        meas, nodes, _branches([]), resolved, telemetry_bounds=bounds, injection_roles=True
    )
    by = {(r["measurement_type"], r.get("source_code")): r for r in rows}
    net_p = by[(4, None)]
    # Generation [10, 400] minus load [50, 200].
    assert (net_p["min_value"], net_p["max_value"]) == pytest.approx((-190.0, 350.0))
    gen = by[(4, SOURCE_INJ_GEN)]
    assert (gen["min_value"], gen["max_value"]) == pytest.approx((10.0, 400.0))
    load = by[(4, SOURCE_INJ_LOAD)]
    assert (load["min_value"], load["max_value"]) == pytest.approx((50.0, 200.0))
    # No component range on Q: the rows carry none.
    assert "min_value" not in by[(5, None)]
    assert "min_value" not in by[(5, SOURCE_INJ_LOAD)]


def test_a_component_without_a_range_unsets_the_side():
    nodes = _nodes([{"id": 5, "voltage_nominal": 220.0}])
    meas = _meas([{"id": 10, "object_type": 0, "object_id": 5, "measurement_type": 2}])
    resolved = {
        (5, "p_gen:0"): (200.0, 1, "g0", GOOD),
        (5, "p_gen:1"): (100.0, 1, "g1", GOOD),
    }
    _, rows = _run(
        meas, nodes, _branches([]), resolved, telemetry_bounds={(5, "p_gen:0"): (0.0, 250.0)}
    )
    assert (rows[0]["min_value"], rows[0]["max_value"]) == (-SENTINEL_VALUE, SENTINEL_VALUE)
