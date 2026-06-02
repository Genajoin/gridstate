"""Unit-тесты ``apply_voltage_meas_calibration_for_gen_nodes``.

V-меры на узлах с активной генерацией + slack получают tight σ².
Pure-load и transit узлы не трогаются.
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import (
    aggregate_generators_to_node,
    apply_voltage_meas_calibration_for_gen_nodes,
)


def _add_voltage_meas(model, node_id: int, value: float, variance: float, mid: int) -> None:
    from gridstate.constants import MeasurementObjectType, MeasurementType

    model.measurements.add(
        {
            "id": mid,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": node_id,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": value,
            "variance": variance,
            "weight": 1.0 / variance,
            "status": True,
            "quality": 0,
        }
    )


def _build_model_three_node_types():
    """3 узла: slack(110), gen-узел(220) с active+off ген, pure-load(110)."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    # 1: slack
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    # 2: gen-узел с одним active и одним off ген
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 220.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PV),
        }
    )
    m.generators.add(
        {
            "id": 21,
            "node_id": 2,
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": True,
        }
    )
    m.generators.add(
        {
            "id": 22,
            "node_id": 2,
            "power_output": 0.0,
            "reactive_output": 0.0,
            "power_min": 0.0,
            "power_max": 60.0,  # игнор
            "reactive_min": -30.0,
            "reactive_max": 40.0,
            "status": False,
        }
    )
    # 3: pure-load
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )

    # V-меры на всех трёх узлах с σ²=50 (типичное baseline)
    _add_voltage_meas(m, 1, 110.5, variance=50.0, mid=1)
    _add_voltage_meas(m, 2, 219.0, variance=50.0, mid=2)
    _add_voltage_meas(m, 3, 109.0, variance=50.0, mid=3)
    return m


def test_calibrates_gen_and_slack_nodes() -> None:
    """gen-узел + slack получают σ²=0.1, pure-load не трогается."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)  # для gen-узла: generation_p_max=100
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 2  # slack=1 + gen=2
    assert stats["updated_meas"] == 2

    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    # Slack: σ² → 0.1
    assert me_by_id[1]["variance"] == pytest.approx(0.1)
    assert me_by_id[1]["weight"] == pytest.approx(10.0)
    # Gen-узел: σ² → 0.1
    assert me_by_id[2]["variance"] == pytest.approx(0.1)
    # Pure-load: σ² не тронут
    assert me_by_id[3]["variance"] == pytest.approx(50.0)


def test_off_only_generator_node_not_calibrated() -> None:
    """Узел только с off-генератором → generation_p_max=0 → не калибруется."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 5,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.generators.add(
        {
            "id": 51,
            "node_id": 5,
            "power_output": 0.0,
            "reactive_output": 0.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": False,  # off
        }
    )
    _add_voltage_meas(m, 5, 109.0, variance=50.0, mid=1)

    aggregate_generators_to_node(m)
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 0
    assert stats["updated_meas"] == 0
    me = m.measurements.to_numpy()
    assert me[0]["variance"] == pytest.approx(50.0)


def test_custom_sigma2_kwarg() -> None:
    """Кастомный σ² через kwarg применяется."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.5)

    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    assert me_by_id[1]["variance"] == pytest.approx(0.5)
    assert me_by_id[1]["weight"] == pytest.approx(2.0)


def test_inactive_measurement_not_touched() -> None:
    """V-мера с status=False не трогается."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    # Деактивируем V-меру на slack
    me_arr = m.measurements.to_numpy().copy()
    me_arr[me_arr["id"] == 1]["status"] = False
    # Через update_from_array применяем
    me_arr["status"][me_arr["id"] == 1] = False
    m.measurements.update_from_array(me_arr)

    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    # Слэк не тронут (status=False)
    assert me_by_id[1]["variance"] == pytest.approx(50.0)
    # Gen-узел тронут
    assert me_by_id[2]["variance"] == pytest.approx(0.1)
    assert stats["updated_meas"] == 1


def test_non_voltage_meas_not_touched() -> None:
    """Не-V меры на gen-узле не трогаются."""
    from gridstate.constants import MeasurementObjectType, MeasurementType

    m = _build_model_three_node_types()
    # P_inj на gen-узле (id=2)
    m.measurements.add(
        {
            "id": 99,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": 2,
            "measurement_type": int(MeasurementType.POWER_INJECTION_P),
            "value": -50.0,
            "variance": 25.0,
            "weight": 0.04,
            "status": True,
            "quality": 0,
        }
    )
    aggregate_generators_to_node(m)
    apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    me = m.measurements.to_numpy()
    p_inj = me[me["id"] == 99][0]
    assert p_inj["variance"] == pytest.approx(25.0)  # не тронут
    assert p_inj["measurement_type"] == int(MeasurementType.POWER_INJECTION_P)


def test_idempotent() -> None:
    """Повторный вызов даёт тот же результат."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    s1 = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)
    s2 = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)
    assert s1 == s2
    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    assert me_by_id[1]["variance"] == pytest.approx(0.1)
    assert me_by_id[2]["variance"] == pytest.approx(0.1)


def test_inactive_node_not_in_targets() -> None:
    """Узел status=False — не калибруется даже если slack/gen."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 7,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": False,  # выключен
            "node_type": int(NodeType.PV),
        }
    )
    m.generators.add(
        {
            "id": 71,
            "node_id": 7,
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": True,
        }
    )
    _add_voltage_meas(m, 7, 109.0, variance=50.0, mid=1)
    aggregate_generators_to_node(m)
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 0
    assert stats["updated_meas"] == 0
