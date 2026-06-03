"""Тесты WLS-алгоритма (``gridstate.algorithms.wls.solve_wls``) и обвязки
``estimate()``.

Подход к синтетике:
    1. Берём произвольную сеть и фиксируем «истинное» состояние ``E_true``
       (V, δ); slack — нулевой угол, V=1.0 p.u.
    2. Через ``BaseAlgebra.evaluate_h`` получаем «идеальные» измерения
       ``z = h(E_true)`` без шума.
    3. Запускаем ``solve_wls`` из flat-старта.
    4. Проверяем ``E_se ≈ E_true``.

Критерий сходимости (≤5 итераций, ``max|ΔE|<1e-6``) проверяется на 3-узловой
сети с переопределённой системой измерений.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.algorithms.wls import solve_wls
from gridstate.api import estimate
from gridstate.state import StateLayout, flat_start, pack, unpack
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    build_z_and_r,
)


# ----------------------------------------------------------- helpers
def _three_bus_for_se():
    """3-узловая сеть. Slack=1; PQ-узлы 2, 3. Достаточно линий для теста."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
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
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
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
            "ti_p_to": 2003,
            "ti_q_to": 2004,
        }
    )
    m.branches.add(
        {
            "id": 300,
            "from_node": 1,
            "to_node": 3,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
            "ti_p_from": 3001,
            "ti_q_from": 3002,
            "ti_p_to": 3003,
            "ti_q_to": 3004,
        }
    )
    return m


def _generate_ideal_measurements(model, e_true: np.ndarray, layout: StateLayout):
    """Сгенерировать коллекцию измерений из ``z = h(E_true)``.

    Покрытие: V на всех узлах, P/Q инъекции на всех узлах, P/Q на «from» каждой
    ветви. Измерения — переопределённая система (m > 2n−1).
    """
    from gridstate.working import Working

    pu = model_to_pu(model)
    ybus, yf, _yt = build_ybus(pu)
    delta_true, v_true = unpack(e_true, layout)

    coll = Working.empty().measurements
    next_id = 1

    # V
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        v_kv = float(v_true[pos] * pu.bus_vn_kv[pos])
        coll.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_VOLTAGE,
                "value": v_kv,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1

    # P_inj, Q_inj — на всех узлах (slack тоже даём, чтобы система была
    # переопределена по углам). В реальных задачах обычно дают P/Q инъекции
    # только на узлах нагрузок/генерации.
    V = v_true * np.exp(1j * delta_true)
    Sbus = V * np.conj(ybus @ V)
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        p_mw = float(Sbus[pos].real * BASE_MVA)
        q_mvar = float(Sbus[pos].imag * BASE_MVA)
        coll.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": p_mw,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        coll.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": q_mvar,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1

    # P_from, Q_from по каждой ветви; ставим ti_p_from / ti_q_from в branch.
    branches_arr = model.branches.to_numpy()
    If = yf @ V
    Sf = V[pu.from_idx] * np.conj(If)
    for i, b in enumerate(branches_arr):
        if not b["status"]:
            continue
        branch_id = int(b["id"])
        p_mw = float(Sf[i].real * BASE_MVA)
        q_mvar = float(Sf[i].imag * BASE_MVA)
        meas_p_id = next_id
        coll.add(
            {
                "id": meas_p_id,
                "object_type": OBJ_BRANCH,
                "object_id": branch_id,
                "measurement_type": KIND_POWER_P,
                "value": p_mw,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        meas_q_id = next_id
        coll.add(
            {
                "id": meas_q_id,
                "object_type": OBJ_BRANCH,
                "object_id": branch_id,
                "measurement_type": KIND_POWER_Q,
                "value": q_mvar,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        # привяжем измерения к ветви через ti_p_from / ti_q_from
        model.branches.update(branch_id, {"ti_p_from": meas_p_id, "ti_q_from": meas_q_id})

    return coll


# ---------------------------------------------------------- basic recovery
def test_solve_wls_recovers_true_state_no_noise() -> None:
    """Без шума WLS должен ровно восстановить ``E_true`` за несколько итераций."""
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)

    rng = np.random.default_rng(0)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.05, 0.05, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.05, 0.05, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)

    coll = _generate_ideal_measurements(m, e_true, layout)
    pu = model_to_pu(m)  # пересоздаём после .update() ветвей
    ybus, yf, yt = build_ybus(pu)
    z, R, idx = build_z_and_r(m, coll, pu)

    e_init = flat_start(layout)
    e_final, success, k, _J = solve_wls(
        e_init,
        z,
        R,
        ybus,
        yf,
        yt,
        idx,
        layout,
        pu,
        tolerance=1e-8,
        max_iterations=20,
    )

    assert success, f"WLS не сошёлся за {k} итераций"
    np.testing.assert_allclose(e_final, e_true, atol=1e-6)


def test_solve_wls_meets_iteration_budget() -> None:
    """Критерий сходимости: WLS сходится за ≤5 итераций при σ=0.01 p.u."""
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)

    rng = np.random.default_rng(123)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.05, 0.05, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.03, 0.03, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)

    coll = _generate_ideal_measurements(m, e_true, layout)
    # Добавим небольшой шум.
    sigma = 0.01
    for meas in coll:
        noise = rng.normal(0.0, sigma * abs(float(meas.value) + 1e-6))
        meas.value = float(meas.value) + noise

    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    z, R, idx = build_z_and_r(m, coll, pu)
    e_init = flat_start(layout)
    e_final, success, k, _J = solve_wls(
        e_init,
        z,
        R,
        ybus,
        yf,
        yt,
        idx,
        layout,
        pu,
        tolerance=1e-6,
        max_iterations=20,
    )
    assert success
    assert k <= 5, f"требуется ≤5 итераций, фактически {k}"
    # Близко к истинному состоянию (разброс ~ σ).
    np.testing.assert_allclose(e_final, e_true, atol=0.05)


# ------------------------------------------------------------- estimate API
def test_estimate_writes_results_back_to_model() -> None:
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)

    rng = np.random.default_rng(7)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.04, 0.04, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.02, 0.02, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)

    coll = _generate_ideal_measurements(m, e_true, layout)
    result = estimate(m, coll, algorithm="wls", init="flat", tolerance=1e-8)
    assert result.success
    assert result.iterations >= 1

    # Проверяем, что V_kV, δ_rad записаны в модель.
    nodes_arr = m.nodes.to_numpy()
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        row = nodes_arr[nodes_arr["id"] == node_id][0]
        v_kv_expected = v_true[pos] * pu.bus_vn_kv[pos]
        assert row["voltage_magnitude"] == pytest.approx(v_kv_expected, abs=1e-4)
        assert row["voltage_angle"] == pytest.approx(delta_true[pos], abs=1e-6)

    # Перетоки/токи на ветвях должны быть проставлены.
    branches_arr = m.branches.to_numpy()
    assert np.all(np.abs(branches_arr["power_from_p"]) > 0)
    assert np.all(np.abs(branches_arr["current_from"]) > 0)


def test_estimate_init_flat_default() -> None:
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    e_true = pack(np.zeros(pu.n_bus), np.ones(pu.n_bus), layout)
    coll = _generate_ideal_measurements(m, e_true, layout)
    # Тривиальный случай: истина = flat → за 1 итерацию.
    result = estimate(m, coll, init="flat", tolerance=1e-10)
    assert result.success
    assert result.iterations <= 2


def test_estimate_init_results_uses_model_state() -> None:
    """init='results' использует voltage_magnitude/angle из модели."""
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    rng = np.random.default_rng(99)
    delta_true = np.zeros(pu.n_bus)
    delta_true[layout.non_slack_idx] = rng.uniform(-0.05, 0.05, size=pu.n_bus - 1)
    v_true = 1.0 + rng.uniform(-0.04, 0.04, size=pu.n_bus)
    e_true = pack(delta_true, v_true, layout)
    coll = _generate_ideal_measurements(m, e_true, layout)

    # Подставляем в model.nodes текущее состояние, близкое к истине.
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        m.nodes.update(
            int(node_id),
            {
                "voltage_magnitude": float(v_true[pos] * pu.bus_vn_kv[pos]),
                "voltage_angle": float(delta_true[pos]),
            },
        )
    result = estimate(m, coll, init="results", tolerance=1e-10)
    assert result.success
    # При init из истины — обязано сойтись быстрее flat-start.
    assert result.iterations <= 3


def test_estimate_unsupported_algorithm_raises() -> None:
    m = _three_bus_for_se()
    with pytest.raises(NotImplementedError, match=r"lav"):
        estimate(m, algorithm="lav")  # type: ignore[arg-type]


def test_estimate_zero_injection_unsupported() -> None:
    m = _three_bus_for_se()
    with pytest.raises(NotImplementedError, match=r"zero_injection"):
        estimate(m, algorithm="wls", zero_injection="aux_bus")


# ------------------------------------------------------------- empty input
def test_solve_wls_empty_measurements_warns() -> None:
    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)
    from gridstate.working import Working

    z, R, idx = build_z_and_r(m, Working.empty().measurements, pu)
    _e_final, success, k, J = solve_wls(flat_start(layout), z, R, ybus, yf, yt, idx, layout, pu)
    assert not success
    assert k == 0
    assert np.isnan(J)


def test_solve_wls_singular_returns_failure() -> None:
    """При сингулярной gain-matrix WLS должен вернуть ``success=False``,
    а не «iter=1, J=NaN, success=True» из-за пустого шага.

    Сценарий: в 3-узловой сети ставим всего одно V-измерение — система
    недоопределена (ранг H ≪ n_state=5). spsolve(G, ...) падает на
    сингулярной матрице, ранний выход не должен интерпретироваться как
    успех.
    """
    from gridstate.working import Working

    m = _three_bus_for_se()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)

    coll = Working.empty().measurements
    coll.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": 110.0,
            "variance": 0.01,
            "status": True,
            "quality": 0,
        }
    )
    z, R, idx = build_z_and_r(m, coll, pu)
    _e_final, success, _k, J = solve_wls(
        flat_start(layout),
        z,
        R,
        ybus,
        yf,
        yt,
        idx,
        layout,
        pu,
        max_iterations=10,
    )
    assert not success, "WLS не должен сообщать успех на сингулярной задаче"
    assert np.isfinite(J), f"J должен быть числом, а не {J}"
