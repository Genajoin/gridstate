"""Per-value variance factor of the producer (``DerivedInputs.measurement_sigma2_scale``).

The factor is the producer's last word on the weight of one telemetry value: it
multiplies the variance of the row the value feeds **after** every built-in rule
of the preparation (QUESTIONABLE multiplier, scale plans, the flow σ floor), and
also a final external variance. A row fed by several values (injection
components) takes the largest factor among them.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.constants import SOURCE_INJ_GEN, SOURCE_INJ_LOAD
from gridstate.contract.derived import DerivedInputs
from gridstate.z_vector import KIND_POWER_INJECTION_P, OBJ_NODE
from tests.test_external_sigma2 import (
    _NO_PASSES,
    _case14,
    _plan,
    _real_active,
    _row_key,
    _run,
    _sigma2_for,
    _stats,
    _two_unit_plan,
)


def _variances(model) -> dict[tuple[int, str], float]:
    return {_row_key(r): float(r["variance"]) for r in _real_active(model)}


def _with_scale(plan: DerivedInputs, scale: dict[tuple[int, str], float]) -> DerivedInputs:
    return DerivedInputs(**{**plan.__dict__, "measurement_sigma2_scale": scale})


def test_factor_applies_after_the_flow_floor():
    model = _case14()
    plan = _plan(model)
    # A floor of 5 % of the channel scale lifts every flow variance of case14.
    base, ev = _run(model, plan, flow_sigma_floor_kv_frac=0.05, **_NO_PASSES)
    assert _stats(ev, "flow_sigma_floor")["floored"] > 0
    var0 = _variances(base)
    flows = [k for k in var0 if k[1] in ("p_from", "q_to")][:2]
    scale = {flows[0]: 0.25, flows[1]: 4.0}
    out, events = _run(model, _with_scale(plan, scale), flow_sigma_floor_kv_frac=0.05, **_NO_PASSES)
    var1 = _variances(out)
    for key, factor in scale.items():
        assert var1[key] == pytest.approx(factor * var0[key]), key
    assert {k: v for k, v in var1.items() if k not in scale} == {
        k: v for k, v in var0.items() if k not in scale
    }
    assert _stats(events, "measurement_sigma2_scale") == {"scaled": 2, "unused": 0}


def test_factor_on_voltage_and_questionable_rows():
    model = _case14()
    plan = _plan(model, questionable=2)
    var0 = _variances(_run(model, plan, **_NO_PASSES)[0])
    v_key = next(k for k in var0 if k[1] == "v")
    q_key = next(
        k for k, v in (plan.telemetry_resolved or {}).items() if v[3] and k[1] != "v" and k in var0
    )
    scale = {v_key: 0.1, q_key: 3.0}
    var1 = _variances(_run(model, _with_scale(plan, scale), **_NO_PASSES)[0])
    for key, factor in scale.items():
        assert var1[key] == pytest.approx(factor * var0[key]), key


def test_factor_multiplies_a_final_external_variance():
    model = _case14()
    plan = _plan(model)
    sigma2 = _sigma2_for(plan)
    plan.measurement_sigma2 = sigma2
    key = next(k for k in sigma2 if k[1] == "p_to")
    out, _ = _run(model, _with_scale(plan, {key: 2.0}), **_NO_PASSES)
    var = _variances(out)
    assert var[key] == pytest.approx(2.0 * sigma2[key])
    assert all(v == sigma2[k] for k, v in var.items() if k != key)


def test_injection_row_takes_the_largest_component_factor():
    model = _case14()
    plan = _two_unit_plan(model)
    node = next(k[0] for k in plan.telemetry_resolved or {} if k[1] == "p_gen:0")
    values = plan.telemetry_resolved or {}
    values[(node, "p_load")] = (5.0, 1, "pl", 0)
    plan.telemetry_arg_keys = list(values)

    def rows(scale):
        out, _ = _run(
            model, _with_scale(plan, scale), algorithm="ipm", ipm_injection_roles=True, **_NO_PASSES
        )
        m = _real_active(out)
        sel = (m["object_type"] == OBJ_NODE) & (m["object_id"] == node)
        sel &= m["measurement_type"] == KIND_POWER_INJECTION_P
        return {str(r["source_code"]): float(r["variance"]) for r in m[sel]}

    base = rows(None)
    got = rows({(node, "p_gen:0"): 2.0, (node, "p_gen:1"): 3.0, (node, "p_load"): 0.5})
    assert got[""] == pytest.approx(3.0 * base[""])  # net row: every component
    assert got[SOURCE_INJ_GEN] == pytest.approx(3.0 * base[SOURCE_INJ_GEN])
    assert got[SOURCE_INJ_LOAD] == pytest.approx(0.5 * base[SOURCE_INJ_LOAD])


def test_without_factors_the_run_is_unchanged():
    model = _case14()
    plan = _plan(model, questionable=2)
    a, ev = _run(model, plan)
    b, _ = _run(model, _with_scale(plan, {}))
    assert a.measurements.to_numpy().tobytes() == b.measurements.to_numpy().tobytes()
    assert a.nodes.to_numpy().tobytes() == b.nodes.to_numpy().tobytes()
    assert "skipped" in _stats(ev, "measurement_sigma2_scale")


def test_unused_keys_are_counted():
    model = _case14()
    plan = _plan(model)
    _, events = _run(model, _with_scale(plan, {(999_999, "v"): 2.0}), **_NO_PASSES)
    assert _stats(events, "measurement_sigma2_scale") == {"scaled": 0, "unused": 1}


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_factor_is_rejected(bad):
    model = _case14()
    plan = _plan(model)
    key = next(iter(plan.telemetry_resolved or {}))
    with pytest.raises(ValueError, match="expected finite > 0"):
        _run(model, _with_scale(plan, {key: bad}))


def test_pseudo_rows_are_not_scaled():
    model = _case14()
    plan = _plan(model)
    node = int(model.nodes.to_numpy()["id"][5])
    plan.pseudo_measurements = {(node, "v"): (140.0, 0.5)}
    out, _ = _run(model, _with_scale(plan, {(node, "v"): 9.0}), **_NO_PASSES)
    m = out.measurements.to_numpy()
    sel = (m["source_code"] == "external_pseudo") & m["status"].astype(bool)
    assert np.asarray(m["variance"][sel]).tolist() == [0.5]
