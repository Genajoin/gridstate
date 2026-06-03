"""Тесты ``gridstate.algebra.base.BaseAlgebra``.

Основная проверка — численная: для случайной сети сравниваем аналитический
якобиан с конечно-разностной аппроксимацией.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.algebra.base import BaseAlgebra
from gridstate.state import StateLayout, flat_start, pack, unpack
from gridstate.units import model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    build_z_and_r,
)


# ---------------------------------------------------------- helpers
def _build_three_bus():
    """3-узловая сеть с двумя ветвями и набором измерений всех типов."""
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
            "generation_p": 60.0,
            "generation_q": 30.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 109.0,
            "voltage_angle": -0.02,
            "load_p": 30.0,
            "load_q": 10.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 108.5,
            "voltage_angle": -0.04,
            "load_p": 30.0,
            "load_q": 20.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 6.05,
            "reactance": 30.25,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 1001,
            "ti_q_from": 1002,
            "ti_p_to": 1003,
            "ti_q_to": 1004,
        }
    )
    m.branches.add(
        {
            "id": 200,
            "from_node": 2,
            "to_node": 3,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 2001,
            "ti_q_from": 2002,
            "ti_p_to": 0,
            "ti_q_to": 0,
        }
    )

    # Измерения: V на всех узлах
    for i, node_id in enumerate([1, 2, 3], start=1):
        m.measurements.add(
            {
                "id": i,
                "object_type": OBJ_NODE,
                "object_id": node_id,
                "measurement_type": KIND_VOLTAGE,
                "value": [110.5, 109.3, 108.8][i - 1],
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
    # P/Q инъекции на не-slack узлах
    m.measurements.add(
        {
            "id": 10,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": -30.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 11,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_POWER_INJECTION_Q,
            "value": -10.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 12,
            "object_type": OBJ_NODE,
            "object_id": 3,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": -30.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    # Перетоки по ветвям (from)
    m.measurements.add(
        {
            "id": 1001,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_P,
            "value": 30.5,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    m.measurements.add(
        {
            "id": 1002,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_Q,
            "value": 12.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    # Переток (to) на ветви 100
    m.measurements.add(
        {
            "id": 1003,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_POWER_P,
            "value": -30.0,
            "variance": 0.5,
            "status": True,
            "quality": 0,
        }
    )
    return m


def _make_algebra(m):
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    z, _R, idx = build_z_and_r(m, m.measurements, pu)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    algebra = BaseAlgebra(ybus, yf, yt, idx, layout, pu)
    return algebra, pu, layout, z, idx


# --------------------------------------------------------- shape / smoke
def test_h_shape_matches_z_length() -> None:
    m = _build_three_bus()
    algebra, _pu, layout, z, _ = _make_algebra(m)
    delta, v = unpack(flat_start(layout), layout)
    h = algebra.evaluate_h(v, delta)
    assert h.shape == z.shape


def test_jacobian_shape() -> None:
    m = _build_three_bus()
    algebra, pu, layout, z, _ = _make_algebra(m)
    delta, v = unpack(flat_start(layout), layout)
    H = algebra.evaluate_jacobian(v, delta)
    assert H.shape == (len(z), 2 * pu.n_bus - 1)


def test_voltage_at_flat_start_equals_one() -> None:
    m = _build_three_bus()
    algebra, _pu, layout, _z, idx = _make_algebra(m)
    delta, v = unpack(flat_start(layout), layout)
    h = algebra.evaluate_h(v, delta)
    v_pos = np.where(idx.kind == KIND_VOLTAGE)[0]
    assert np.allclose(h[v_pos], 1.0)


# ------------------------------------------- finite-difference Jacobian check
def _numeric_jacobian(
    algebra: BaseAlgebra, layout: StateLayout, e: np.ndarray, eps: float = 1e-7
) -> np.ndarray:
    delta0, v0 = unpack(e, layout)
    h0 = algebra.evaluate_h(v0, delta0)
    m = h0.shape[0]
    n_state = e.shape[0]
    J = np.zeros((m, n_state), dtype=np.float64)
    for j in range(n_state):
        e_pert = e.copy()
        e_pert[j] += eps
        delta_p, v_p = unpack(e_pert, layout)
        h_p = algebra.evaluate_h(v_p, delta_p)
        J[:, j] = (h_p - h0) / eps
    return J


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_jacobian_matches_finite_differences(seed: int) -> None:
    """Аналитический якобиан совпадает с численным (FD)."""
    m = _build_three_bus()
    algebra, pu, layout, _z, _ = _make_algebra(m)
    rng = np.random.default_rng(seed)
    # Нерасплющенное состояние, чтобы избежать сингулярности |I|=0.
    delta = rng.uniform(-0.05, 0.05, size=pu.n_bus)
    delta[layout.slack_idx] = 0.0
    v = 1.0 + rng.uniform(-0.05, 0.05, size=pu.n_bus)
    e = pack(delta, v, layout)

    H_analytic = algebra.evaluate_jacobian(v, delta).toarray()
    H_numeric = _numeric_jacobian(algebra, layout, e, eps=1e-7)
    np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-4, rtol=1e-4)


def test_h_voltage_in_kV_correctly_normalized() -> None:
    """V-измерение vn=110, V_pu=1.0 → ``h = 1.0`` (а не 110)."""
    m = _build_three_bus()
    algebra, pu, _layout, _z, idx = _make_algebra(m)
    delta = np.zeros(pu.n_bus)
    v = np.ones(pu.n_bus)
    h = algebra.evaluate_h(v, delta)
    v_pos = np.where(idx.kind == KIND_VOLTAGE)[0]
    assert np.allclose(h[v_pos], 1.0)


# --------------------------------------------------- branch power direction
def test_branch_power_to_side_uses_to_node_voltage() -> None:
    """h(POWER_P, side=TO) ≠ h(POWER_P, side=FROM) при ненулевом перетоке."""
    m = _build_three_bus()
    algebra, _pu, _layout, _z, idx = _make_algebra(m)
    # Состояние не-flat — нужна разница углов, чтобы был переток.
    delta = np.array([0.0, -0.05, -0.10])
    v = np.array([1.0, 0.98, 0.95])
    h = algebra.evaluate_h(v, delta)
    p_from = float(h[idx.meas_id == 1001].item())
    p_to = float(h[idx.meas_id == 1003].item())
    # Активный поток: |P_from + P_to| = потери R·I² ≪ |P_from|.
    assert p_from > 0  # шлёт от 1 к 2
    assert p_to < 0  # узел 2 принимает
    assert abs(p_from + p_to) < 0.05 * abs(p_from)  # потери небольшие


# ------------------------------------------------ degenerate / empty inputs
def test_empty_measurements_returns_zero_jacobian() -> None:
    m = _build_three_bus()
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    # Пустой MeasurementCollection
    from gridstate.working import Working

    empty = Working.empty().measurements
    _z, _R, idx = build_z_and_r(m, empty, pu)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    algebra = BaseAlgebra(ybus, yf, yt, idx, layout, pu)
    delta = np.zeros(pu.n_bus)
    v = np.ones(pu.n_bus)
    H = algebra.evaluate_jacobian(v, delta)
    assert H.shape == (0, 2 * pu.n_bus - 1)
    h = algebra.evaluate_h(v, delta)
    assert h.shape == (0,)


def test_balance_meas_h_uses_box_vars() -> None:
    """Balance-meas: h[i] = Sbus[i] - (Pgen − Pnag) / (Qgen − Qnag)."""
    from gridstate.z_vector import (
        KIND_NODE_BALANCE_P,
        KIND_NODE_BALANCE_Q,
        SIDE_NONE,
        MeasurementIndex,
    )

    m = _build_three_bus()
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    # Layout с box-vars на узле 2 (позиция в bus_ids):
    pos_node2 = int(np.where(pu.bus_ids == 2)[0][0])
    layout = StateLayout(
        n_bus=pu.n_bus,
        slack_idx=pu.slack_idx,
        non_slack_idx=np.array([i for i in range(pu.n_bus) if i != pu.slack_idx], dtype=np.int64),
        pgen_node_pos=np.array([pos_node2], dtype=np.int64),
        qgen_node_pos=np.array([pos_node2], dtype=np.int64),
        pnag_node_pos=np.array([pos_node2], dtype=np.int64),
        qnag_node_pos=np.array([pos_node2], dtype=np.int64),
    )
    # Создаём MeasurementIndex с двумя balance-meas (P, Q) на узле 2.
    idx = MeasurementIndex(
        kind=np.array([KIND_NODE_BALANCE_P, KIND_NODE_BALANCE_Q], dtype=np.int64),
        object_kind=np.array([OBJ_NODE, OBJ_NODE], dtype=np.int64),
        object_pos=np.array([pos_node2, pos_node2], dtype=np.int64),
        branch_side=np.array([SIDE_NONE, SIDE_NONE], dtype=np.int64),
        meas_id=np.array([9000, 9001], dtype=np.int64),
    )
    algebra = BaseAlgebra(ybus, yf, yt, idx, layout, pu)

    v = np.array([1.0, 0.99, 0.985])
    delta = np.array([0.0, -0.02, -0.04])

    # Без box-values (None) — balance принимает их как 0:
    # h_balance_P = Sbus.real - 0 = Sbus.real
    h_no_box = algebra.evaluate_h(v, delta)

    # С box-values: pgen=0.6, pnag=0.3 → -(0.6-0.3) = -0.3
    h_with_box = algebra.evaluate_h(
        v,
        delta,
        pgen_estimated=np.array([0.6]),
        qgen_estimated=np.array([0.0]),
        pnag_estimated=np.array([0.3]),
        qnag_estimated=np.array([0.0]),
    )
    # h_with_box[0] (P-balance) = Sbus.real − 0.3 = h_no_box[0] − 0.3
    assert h_with_box[0] == pytest.approx(h_no_box[0] - 0.3, abs=1e-9)
    # Q: pgen=qnag=0 → разница 0
    assert h_with_box[1] == pytest.approx(h_no_box[1], abs=1e-9)


def test_balance_meas_jacobian_box_columns() -> None:
    """Balance jacobian: ±1 в нужных box-столбцах, размер с учётом IPM."""
    from gridstate.z_vector import (
        KIND_NODE_BALANCE_P,
        SIDE_NONE,
        MeasurementIndex,
    )

    m = _build_three_bus()
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    pos_node2 = int(np.where(pu.bus_ids == 2)[0][0])
    layout = StateLayout(
        n_bus=pu.n_bus,
        slack_idx=pu.slack_idx,
        non_slack_idx=np.array([i for i in range(pu.n_bus) if i != pu.slack_idx], dtype=np.int64),
        pgen_node_pos=np.array([pos_node2], dtype=np.int64),
        pnag_node_pos=np.array([pos_node2], dtype=np.int64),
    )
    idx = MeasurementIndex(
        kind=np.array([KIND_NODE_BALANCE_P], dtype=np.int64),
        object_kind=np.array([OBJ_NODE], dtype=np.int64),
        object_pos=np.array([pos_node2], dtype=np.int64),
        branch_side=np.array([SIDE_NONE], dtype=np.int64),
        meas_id=np.array([9000], dtype=np.int64),
    )
    algebra = BaseAlgebra(ybus, yf, yt, idx, layout, pu)
    v = np.ones(pu.n_bus)
    delta = np.zeros(pu.n_bus)
    H = algebra.evaluate_jacobian(v, delta).toarray()
    # Размер: (1, 2*n_bus - 1 + 2)  (n_box = 2 — Pgen + Pnag для одного узла)
    assert H.shape == (1, 2 * pu.n_bus - 1 + 2)
    # Последние 2 колонки — box-vars (Pgen, Pnag по offset_pgen/pnag).
    # offset_pgen = 2*n_bus - 1, offset_pnag = 2*n_bus - 1 + 1 (qgen, qnag пусты).
    # Проверяем: -1 в pgen-столбце, +1 в pnag-столбце.
    assert H[0, 2 * pu.n_bus - 1] == -1.0  # ∂h/∂Pgen[node2] = -1
    assert H[0, 2 * pu.n_bus - 1 + 1] == +1.0  # ∂h/∂Pnag[node2] = +1


def test_current_measurement_jacobian_with_finite_difference() -> None:
    """Якобиан |I| ветви — самый коварный из-за ``conj(I)/|I|``-сингулярности."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.SLACK)}
    )
    m.nodes.add({"id": 2, "voltage_nominal": 110.0, "status": True, "node_type": int(NodeType.PQ)})
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 6.05,
            "reactance": 30.25,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 9001,
            "ti_q_from": 9002,
        }
    )
    m.measurements.add(
        {
            "id": 9001,
            "object_type": OBJ_BRANCH,
            "object_id": 100,
            "measurement_type": KIND_CURRENT,
            "value": 50.0,
            "variance": 1.0,
            "status": True,
            "quality": 0,
        }
    )
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    _z, _R, idx = build_z_and_r(m, m.measurements, pu)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    algebra = BaseAlgebra(ybus, yf, yt, idx, layout, pu)
    # Ненулевой ток — небольшой перепад
    delta = np.array([0.0, -0.03])
    v = np.array([1.0, 0.97])
    e = pack(delta, v, layout)
    H_analytic = algebra.evaluate_jacobian(v, delta).toarray()
    H_numeric = _numeric_jacobian(algebra, layout, e, eps=1e-7)
    np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-4, rtol=1e-3)
