"""Тесты fallback-логики ``apply_voltage_range_filter``.

Покрывают сценарии заглушек входного формата (`U_KRIT=1.0` на 500-кВ узлах,
отсутствие `U_MAX` и т.п.).
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


def test_u_krit_stub_falls_through_to_u_min() -> None:
    """U_KRIT=1.0 — заглушка эталонной SE; должны взять U_MIN=350 как нижнюю границу."""
    m = _build({"voltage_critical": 1.0, "voltage_min": 350.0}, v_meas=400.0)
    stats = apply_voltage_range_filter(m)
    me = m.measurements.to_numpy()
    # 400 > 350 → V в диапазоне → status=True
    assert bool(me[0]["status"]) is True
    assert me[0]["variance"] == pytest.approx(100.0)
    assert stats["out_of_range"] == 0


def test_u_krit_stub_v_below_u_min_downweight() -> None:
    """V=300 < U_MIN=350 → out-of-range при заглушке U_KRIT."""
    m = _build({"voltage_critical": 1.0, "voltage_min": 350.0}, v_meas=300.0)
    stats = apply_voltage_range_filter(m)
    me = m.measurements.to_numpy()
    assert me[0]["variance"] == pytest.approx(100.0 * 100.0)  # downweight ×100
    assert stats["out_of_range"] == 1


def test_both_u_krit_and_u_min_stubs_use_half_vnom() -> None:
    """Обе заглушки (U_KRIT=1.0, U_MIN=1.0) → fallback V_ном/2 = 250."""
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


def test_u_krit_valid_used_directly() -> None:
    """Валидный U_KRIT=350 (≥ V_ном/2) — берём как нижнюю границу."""
    m = _build({"voltage_critical": 350.0, "voltage_min": 0.0}, v_meas=349.0)
    stats = apply_voltage_range_filter(m)
    assert stats["out_of_range"] == 1  # 349 < 350


def test_u_max_zero_uses_factor_fallback() -> None:
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


def test_u_max_set_takes_precedence() -> None:
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
