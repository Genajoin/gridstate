"""Measurement noise model (:mod:`gridstate.telemetry.noise`)."""

from __future__ import annotations

import math

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.telemetry import noise, units


def test_units_module_reexports_the_noise_functions() -> None:
    for name in ("variance_voltage", "variance_power", "variance_branch_q"):
        assert getattr(units, name) is getattr(noise, name)
    assert units.normalize_guid("{AB-cd}") == "ab-cd"


def test_variance_injection_matches_former_inline_formula() -> None:
    for net in (0.0, 3.0, -7.5, 10.0, 123.456, -2500.0):
        for pq, base in (("P", 1.0), ("Q", 0.5)):
            sigma = max(0.05 * abs(net), 0.5)
            expected = sigma * sigma + base
            got = noise.variance_injection(net, pq)
            assert got == expected  # bit-exact
            assert type(got) is float


def test_uniform_box_variance() -> None:
    assert noise.uniform_box_variance(-6.0, 6.0) == 12.0


def test_gen_v_calibrated_variance() -> None:
    s2 = noise.GEN_V_SIGMA2
    assert noise.gen_v_calibrated_variance(0.01, s2, tighten=False) == s2
    assert noise.gen_v_calibrated_variance(5.0, s2, tighten=False) == s2
    assert noise.gen_v_calibrated_variance(0.01, s2, tighten=True) is None
    assert noise.gen_v_calibrated_variance(s2, s2, tighten=True) is None
    assert noise.gen_v_calibrated_variance(5.0, s2, tighten=True) == s2
    assert noise.gen_v_calibrated_variance(0.0, s2, tighten=True) == s2
    assert noise.gen_v_calibrated_variance(math.nan, s2, tighten=True) == s2


def test_flow_sigma_floor() -> None:
    vn = np.array([0.0, 110.0, 500.0])
    floor = noise.flow_sigma_floor(vn, kv_frac=0.01, current_ka=1.0)
    np.testing.assert_array_equal(floor, (0.01 * np.sqrt(3.0) * 1.0) * vn)
    assert math.isclose(floor[1], 1.905, rel_tol=1e-3)


def test_branch_nominal_kv_policies() -> None:
    br = np.zeros(2, dtype=SE_INPUT.branches.input_dtype())
    br["id"] = [10, 11]
    br["from_node"] = [1, 3]
    br["to_node"] = [2, 99]
    vn = {1: 110.0, 2: 220.0, 3: 35.0}
    assert noise.branch_nominal_kv(vn, br, policy="from") == {10: 110.0, 11: 35.0}
    assert noise.branch_nominal_kv(vn, br, policy="max") == {10: 220.0, 11: 35.0}
