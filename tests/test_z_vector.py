"""Тесты сборки вектора измерений (``gridstate.z_vector``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.units import BASE_MVA, model_to_pu
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    SIDE_FROM,
    SIDE_TO,
    build_z_and_r,
)


# ----------------------------------------------------------- helpers
def _build_two_bus_with_measurements():
    """Сеть из 2 узлов и 1 ветви + набор измерений всех типов."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 0.0,
            "load_q": 0.0,
            "generation_p": 50.0,
            "generation_q": 20.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 30.0,
            "load_q": 10.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )

    # Связь измерений ↔ ветви через ti_*-поля
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 1001,  # связь с измерением id=1001
            "ti_q_from": 1002,
            "ti_p_to": 1003,
            "ti_q_to": 1004,
        }
    )

    # Узловые измерения
    m.measurements.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": 110.5,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": 109.7,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 3,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 50.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 4,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_Q,
            "value": -10.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
        }
    )
    # Ветвевые измерения с известными side
    m.measurements.add(
        {
            "id": 1001,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_P,
            "value": 28.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 1003,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_P,
            "value": -28.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    return m


# --------------------------------------------------------------- basic
def test_basic_assembly_lengths_consistent() -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    z, R, idx = build_z_and_r(m, m.measurements, pu)
    assert len(z) == 6
    assert R.shape == (6, 6)
    assert len(idx) == 6


# ------------------------------------------------------- unit conversions
class TestUnitConversion:
    def test_voltage_kv_to_pu(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        v_pos = np.where(idx.kind == KIND_VOLTAGE)[0]
        # Узлы 1 и 2: vn=110 кВ, value=110.5 → 1.00454
        for i in v_pos:
            assert 0.99 < z[i] < 1.01

    def test_voltage_variance_scaled(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        _, R, idx = build_z_and_r(m, m.measurements, pu)
        v_pos = np.where(idx.kind == KIND_VOLTAGE)[0][0]
        # variance=0.01 кВ², vn=110 → σ²_pu = 0.01 / 110² ≈ 8.26e-7
        assert R.diagonal()[v_pos] == pytest.approx(0.01 / (110.0**2))

    def test_power_mw_to_pu(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        # POWER_INJECTION_P=50 МВт → 0.5 p.u.
        p_inj_pos = np.where(idx.kind == KIND_POWER_INJECTION_P)[0][0]
        assert z[p_inj_pos] == pytest.approx(50.0 / BASE_MVA)
        # Перетоковая POWER_P (id=1001, value=28) → 0.28 p.u.
        p_branch = np.where((idx.kind == KIND_POWER_P) & (idx.branch_side == SIDE_FROM))[0][0]
        assert z[p_branch] == pytest.approx(28.0 / BASE_MVA)


# -------------------------------------------------------- side detection
class TestBranchSide:
    def test_from_and_to_resolved_via_ti_fields(self) -> None:
        m = _build_two_bus_with_measurements()
        pu = model_to_pu(m)
        _, _, idx = build_z_and_r(m, m.measurements, pu)
        # id=1001 ↔ ti_p_from → SIDE_FROM
        # id=1003 ↔ ti_p_to → SIDE_TO
        for k, mid in zip(idx.kind, idx.meas_id, strict=False):
            if mid == 1001:
                assert k == KIND_POWER_P
            if mid == 1003:
                assert k == KIND_POWER_P
        side_for = {int(mid): int(s) for mid, s in zip(idx.meas_id, idx.branch_side, strict=False)}
        assert side_for[1001] == SIDE_FROM
        assert side_for[1003] == SIDE_TO

    def test_unresolvable_side_skipped(self, caplog) -> None:
        m = _build_two_bus_with_measurements()
        # Добавим ветвевое измерение, не привязанное ни к одному ti_*
        m.measurements.add(
            {
                "id": 9999,
                "object_type": OBJ_BRANCH,
                "object_id": 100,
                "measurement_type": KIND_POWER_P,
                "value": 0.0,
                "variance": 1.0,
                "status": True,
                "quality": 0,
            }
        )
        pu = model_to_pu(m)
        with caplog.at_level("WARNING"):
            _, _, idx = build_z_and_r(m, m.measurements, pu)
        # Должно прийти 6 валидных + 0 невалидных
        assert 9999 not in idx.meas_id.tolist()


# ------------------------------------------------------- filtering
class TestFiltering:
    def test_inactive_status_skipped(self) -> None:
        m = _build_two_bus_with_measurements()
        # Помечаем измерение id=1 как неактивное.
        m.measurements.get_by_id(1).status = False
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        assert 1 not in idx.meas_id.tolist()
        assert len(z) == 5

    def test_bad_quality_skipped(self) -> None:
        m = _build_two_bus_with_measurements()
        m.measurements.get_by_id(2).quality = 2  # BAD
        pu = model_to_pu(m)
        z, _, idx = build_z_and_r(m, m.measurements, pu)
        assert 2 not in idx.meas_id.tolist()
        assert len(z) == 5

    def test_negative_variance_skipped(self) -> None:
        m = _build_two_bus_with_measurements()
        m.measurements.get_by_id(1).variance = -0.1
        pu = model_to_pu(m)
        _z, _, idx = build_z_and_r(m, m.measurements, pu)
        assert 1 not in idx.meas_id.tolist()


# ------------------------------------------------------- index integrity
def test_meas_index_uses_positional_indices() -> None:
    m = _build_two_bus_with_measurements()
    pu = model_to_pu(m)
    _, _, idx = build_z_and_r(m, m.measurements, pu)
    # Узловые измерения должны иметь object_pos в [0, n_bus)
    for k, op in zip(idx.object_kind, idx.object_pos, strict=False):
        if k == OBJ_NODE:
            assert 0 <= op < pu.n_bus
        elif k == OBJ_BRANCH:
            assert 0 <= op < pu.n_branch


# --------------------------------------------------------- empty input
def test_empty_collection_returns_zeros() -> None:
    """Сеть без измерений: z пуст, R 0×0."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    pu = model_to_pu(m)
    z, R, idx = build_z_and_r(m, m.measurements, pu)
    assert len(z) == 0
    assert R.shape == (0, 0)
    assert len(idx) == 0


# --------------------------------------------------- unknown bus filtered
def test_measurement_on_unknown_node_skipped() -> None:
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    m.measurements.add(
        {
            "id": 5,
            "object_type": OBJ_NODE,
            "object_id": 999,  # узла 999 нет
            "measurement_type": KIND_VOLTAGE,
            "value": 110.0,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    pu = model_to_pu(m)
    z, _, idx = build_z_and_r(m, m.measurements, pu)
    assert len(z) == 0
    assert 5 not in idx.meas_id.tolist()
