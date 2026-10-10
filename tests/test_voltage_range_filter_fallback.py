"""Тесты fallback-логики ``apply_voltage_range_filter``.

They cover placeholder limits found in real inputs (a critical limit of 1.0 on
500 kV nodes, a missing upper limit and the like).
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import apply_voltage_range_filter


def _add_v_meas(model, node_id: int, value: float, mid: int) -> None:
    from gridstate.constants import MeasurementObjectType, MeasurementType

    model.measurements.add(
        {
            "id": mid,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": node_id,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": value,
            "variance": 100.0,
            "weight": 0.01,
            "status": True,
            "quality": 0,
        }
    )


def _build(node_attrs: dict, v_meas: float):
    """Минимальная модель: 1 активный узел + 1 V-мера."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    base = {
        "id": 1,
        "voltage_nominal": 500.0,
        "status": True,
        "node_type": int(NodeType.PQ),
    }
    base.update(node_attrs)
    m.nodes.add(base)
    _add_v_meas(m, 1, v_meas, mid=1)
    return m


def test_critical_placeholder_falls_through_to_min_limit() -> None:
    """A critical limit of 1.0 is a placeholder; voltage_min=350 is the lower bound."""
    m = _build({"voltage_critical": 1.0, "voltage_min": 350.0}, v_meas=400.0)
    stats = apply_voltage_range_filter(m)
    me = m.measurements.to_numpy()
    # 400 > 350 → V в диапазоне → status=True
    assert bool(me[0]["status"]) is True
    assert me[0]["variance"] == pytest.approx(100.0)
    assert stats["out_of_range"] == 0


def test_critical_placeholder_v_below_min_limit_downweighted() -> None:
    """V=300 < voltage_min=350 → out of range with a placeholder critical limit."""
    m = _build({"voltage_critical": 1.0, "voltage_min": 350.0}, v_meas=300.0)
    stats = apply_voltage_range_filter(m)
    me = m.measurements.to_numpy()
    assert me[0]["variance"] == pytest.approx(100.0 * 100.0)  # downweight ×100
    assert stats["out_of_range"] == 1


def test_critical_and_min_placeholders_use_half_vnom() -> None:
    """Both placeholders (critical 1.0, min 1.0) → fallback V_nom/2 = 250."""
    m = _build({"voltage_critical": 1.0, "voltage_min": 1.0}, v_meas=200.0)
    stats = apply_voltage_range_filter(m)
    m.measurements.to_numpy()
    # 200 < 250 (=V_ном/2) → out-of-range
    assert stats["out_of_range"] == 1
    # 260 > 250 → ok
    m2 = _build({"voltage_critical": 1.0, "voltage_min": 1.0}, v_meas=260.0)
    apply_voltage_range_filter(m2)
    me2 = m2.measurements.to_numpy()
    assert bool(me2[0]["status"]) is True


def test_valid_critical_limit_used_directly() -> None:
    """A valid critical limit of 350 (≥ V_nom/2) is the lower bound."""
    m = _build({"voltage_critical": 350.0, "voltage_min": 0.0}, v_meas=349.0)
    stats = apply_voltage_range_filter(m)
    assert stats["out_of_range"] == 1  # 349 < 350


def test_missing_upper_limit_uses_factor_fallback() -> None:
    """voltage_max=0 → fallback 1.4·V_ном = 700."""
    m = _build(
        {"voltage_critical": 350.0, "voltage_min": 350.0, "voltage_max": 0.0},
        v_meas=750.0,
    )
    stats = apply_voltage_range_filter(m)
    assert stats["out_of_range"] == 1  # 750 > 700
    m2 = _build(
        {"voltage_critical": 350.0, "voltage_min": 350.0, "voltage_max": 0.0},
        v_meas=650.0,
    )
    apply_voltage_range_filter(m2)
    assert bool(m2.measurements.to_numpy()[0]["status"]) is True  # 650 < 700


def test_upper_limit_set_takes_precedence() -> None:
    """voltage_max=525 → hi = 525·1.10 = 577.5."""
    m = _build(
        {"voltage_critical": 350.0, "voltage_min": 350.0, "voltage_max": 525.0},
        v_meas=600.0,
    )
    stats = apply_voltage_range_filter(m)
    assert stats["out_of_range"] == 1  # 600 > 577.5
    m2 = _build(
        {"voltage_critical": 350.0, "voltage_min": 350.0, "voltage_max": 525.0},
        v_meas=550.0,
    )
    apply_voltage_range_filter(m2)
    assert bool(m2.measurements.to_numpy()[0]["status"]) is True  # 550 < 577.5


def test_custom_upper_fallback_factor() -> None:
    """Передача upper_fallback_factor меняет верхнюю границу."""
    m = _build(
        {"voltage_critical": 350.0, "voltage_min": 350.0, "voltage_max": 0.0},
        v_meas=750.0,
    )
    apply_voltage_range_filter(m, upper_fallback_factor=1.6)  # hi = 800
    assert bool(m.measurements.to_numpy()[0]["status"]) is True  # 750 < 800


def test_deactivate_action() -> None:
    """action=deactivate ставит status=False вместо downweight."""
    m = _build({"voltage_critical": 350.0, "voltage_min": 350.0}, v_meas=300.0)
    apply_voltage_range_filter(m, action="deactivate")
    me = m.measurements.to_numpy()
    assert bool(me[0]["status"]) is False


def test_min_voltage_nominal_kv_default_skips_lv() -> None:
    """Дефолт min_voltage_nominal_kv=110: LV-узел (11 кВ) не проверяется вовсе."""
    m = _build({"voltage_nominal": 11.0, "voltage_critical": 0.0}, v_meas=21.2)
    stats = apply_voltage_range_filter(m)
    me = m.measurements.to_numpy()
    # 21.2 кВ на 11 кВ шине (1.93 pu) — мусор, но класс < 110 кВ пропущен
    assert bool(me[0]["status"]) is True
    assert me[0]["variance"] == pytest.approx(100.0)
    assert stats["checked"] == 0


def test_min_voltage_nominal_kv_zero_checks_lv() -> None:
    """min_voltage_nominal_kv=0: LV-класс проверяется — датчик чужого класса ×~2
    (реальный кейс СК-2: 21.2 кВ на 11 кВ шине) downweight'ится."""
    m = _build({"voltage_nominal": 11.0, "voltage_critical": 0.0}, v_meas=21.2)
    stats = apply_voltage_range_filter(m, min_voltage_nominal_kv=0.0)
    me = m.measurements.to_numpy()
    # hi = 11·1.4 = 15.4 < 21.2 → out_of_range → downweight σ²×100
    assert stats["out_of_range"] == 1
    assert me[0]["variance"] == pytest.approx(100.0 * 100.0)
    # валидная LV-мера при этом проходит
    m2 = _build({"voltage_nominal": 11.0, "voltage_critical": 0.0}, v_meas=10.8)
    stats2 = apply_voltage_range_filter(m2, min_voltage_nominal_kv=0.0)
    assert stats2["out_of_range"] == 0
    assert bool(m2.measurements.to_numpy()[0]["status"]) is True
