"""Тесты обнаружения и удаления плохих данных.

Покрывает ``gridstate.validation.chi2_test`` и ``gridstate.validation.bad_data``.

Сценарии:
    1. Чистые данные → χ²-тест не срабатывает;
    2. Инъекция большого выброса → ``chi2_analysis.bad_data_present == True``,
       ``remove_bad_data`` помечает именно это измерение как BAD;
    3. После удаления плохого измерения χ²-тест проходит.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.validation.bad_data import QUALITY_BAD, remove_bad_data
from gridstate.validation.chi2_test import Chi2Result, chi2_analysis
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


# ----------------------------------------------------------- helpers
def _three_bus_with_clean_measurements():
    """Та же 3-узловая сеть, что в test_wls; меры из истинного состояния."""
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
    # «Истинное» состояние: фиксируем небольшие нерасплющенные значения.
    delta_true = np.array([0.0, -0.04, -0.06])
    v_true = np.array([1.0, 0.98, 0.97])
    _add_synthetic_measurements(m, delta_true, v_true)
    return m, delta_true, v_true


def _add_synthetic_measurements(model, delta_true, v_true):
    from gridstate.state import StateLayout
    from gridstate.ybus import build_ybus

    pu = model_to_pu(model)
    ybus, yf, _yt = build_ybus(pu)
    StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    V = v_true * np.exp(1j * delta_true)
    Sbus = V * np.conj(ybus @ V)
    If = yf @ V
    Sf = V[pu.from_idx] * np.conj(If)

    next_id = 1
    # V на всех узлах
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        v_kv = float(v_true[pos] * pu.bus_vn_kv[pos])
        model.measurements.add(
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
    # P_inj/Q_inj на всех узлах
    for pos, node_id in enumerate(pu.bus_ids.tolist()):
        model.measurements.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": float(Sbus[pos].real * BASE_MVA),
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        model.measurements.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": int(node_id),
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": float(Sbus[pos].imag * BASE_MVA),
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
    # P/Q «from» каждой ветви + ti_*_from
    branches_arr = model.branches.to_numpy()
    for i, b in enumerate(branches_arr):
        if not b["status"]:
            continue
        branch_id = int(b["id"])
        meas_p_id = next_id
        model.measurements.add(
            {
                "id": meas_p_id,
                "object_type": OBJ_BRANCH,
                "object_id": branch_id,
                "measurement_type": KIND_POWER_P,
                "value": float(Sf[i].real * BASE_MVA),
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        meas_q_id = next_id
        model.measurements.add(
            {
                "id": meas_q_id,
                "object_type": OBJ_BRANCH,
                "object_id": branch_id,
                "measurement_type": KIND_POWER_Q,
                "value": float(Sf[i].imag * BASE_MVA),
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        model.branches.update(
            branch_id,
            {
                "ti_p_from": meas_p_id,
                "ti_q_from": meas_q_id,
            },
        )


# ------------------------------------------------------------- chi2 tests
def test_chi2_passes_on_clean_data() -> None:
    """Без шума и без выбросов: J ≈ 0 → bad_data_present == False."""
    m, _, _ = _three_bus_with_clean_measurements()
    estimate(m, tolerance=1e-10)
    res = chi2_analysis(m)
    assert isinstance(res, Chi2Result)
    assert res.degrees_of_freedom > 0
    assert not res.bad_data_present
    assert res.objective < res.threshold


def test_chi2_detects_large_outlier() -> None:
    """Один V-выброс на узле → χ² ловит."""
    m, _, _ = _three_bus_with_clean_measurements()
    # Подменяем одно V на 130 кВ (вместо ~108) — гигантский выброс.
    m.measurements.get_by_id(2).value = 130.0
    estimate(m, tolerance=1e-10)
    res = chi2_analysis(m, chi2_prob_false=0.01)
    assert res.bad_data_present
    assert res.objective > res.threshold


def test_chi2_invalid_alpha_raises() -> None:
    m, _, _ = _three_bus_with_clean_measurements()
    estimate(m)
    with pytest.raises(ValueError, match=r"chi2_prob_false"):
        chi2_analysis(m, chi2_prob_false=1.5)


def test_chi2_warns_when_df_nonpositive(caplog) -> None:
    """Когда m == n−1, df<0 → возвращаем bad_data_present=False с warning."""
    from gridstate.constants import NodeType
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
    # Только одно измерение → m=1, n_state=3, df=−2
    m.measurements.add(
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
    with caplog.at_level("WARNING"):
        res = chi2_analysis(m)
    assert res.degrees_of_freedom < 0
    assert not res.bad_data_present
    assert "df = m" in caplog.text


# ------------------------------------------------------------- bad_data tests
def test_remove_bad_data_marks_outlier_as_bad() -> None:
    """5σ-выброс на P-инъекции узла 2 должен быть удалён."""
    m, _, _ = _three_bus_with_clean_measurements()
    # Найдём id измерения P_INJ для node 2: это второе сгенерированное P_inj.
    # Смотрим в коллекции по object_id и measurement_type.
    bad_meas = next(
        meas
        for meas in m.measurements
        if int(meas.object_type) == OBJ_NODE
        and int(meas.object_id) == 2
        and int(meas.measurement_type) == KIND_POWER_INJECTION_P
    )
    bad_id = int(bad_meas.id)
    bad_meas.value = float(bad_meas.value) + 50.0  # ~70σ выброс при σ²=0.5

    result = remove_bad_data(m, rn_max_threshold=3.0, max_iterations=5)
    assert result.converged
    assert bad_id in result.removed_meas_ids
    # Соответствующее измерение должно быть BAD/inactive.
    flagged = m.measurements.get_by_id(bad_id)
    assert flagged.status is False or bool(flagged.status) is False
    assert int(flagged.quality) == QUALITY_BAD


def test_remove_bad_data_clean_data_no_removals() -> None:
    """На чистых данных rn_max-тест проходит сразу, ничего не удаляется."""
    m, _, _ = _three_bus_with_clean_measurements()
    result = remove_bad_data(m)
    assert result.converged
    assert result.removed_meas_ids == []
    assert len(result.rn_max_history) == 1
    assert result.rn_max_history[0] < 3.0


def test_chi2_passes_after_bad_data_removal() -> None:
    """После удаления выброса χ² должен снова пройти."""
    m, _, _ = _three_bus_with_clean_measurements()
    bad_meas = next(
        meas
        for meas in m.measurements
        if int(meas.object_type) == OBJ_NODE
        and int(meas.object_id) == 3
        and int(meas.measurement_type) == KIND_POWER_INJECTION_Q
    )
    bad_meas.value = float(bad_meas.value) + 80.0

    result = remove_bad_data(m, rn_max_threshold=3.0)
    assert result.converged
    assert len(result.removed_meas_ids) >= 1
    chi2_res = chi2_analysis(m)
    assert not chi2_res.bad_data_present
