"""Транзитные узлы в IPM: жёсткий zero-injection через transit_balance_sigma2_pu.

Baseline-поведение (флаг OFF): balance-строки транзитных узлов
(``exist_load=0`` и ``exist_gen=0``) получают ту же мягкую адаптивную σ²,
что и все узлы; солвер оставляет на транзите остаточную инжекцию, и
``reconcile_node_balance`` материализует её псевдонагрузкой.

С флагом (>0): balance-строки транзита становятся tight virtual-measurement
нулевой инжекции — состояние обязано удовлетворять KCL транзита, паразитная
псевдонагрузка исчезает. Slack из затяжки исключён (закрывает потери).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.api import estimate
from gridstate.constants import BranchType, NodeType
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.state import StateLayout
from gridstate.units import model_to_pu
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_VOLTAGE,
    OBJ_NODE,
    MeasurementIndex,
)


def _build_line(m: Working, bid: int, f: int, t: int) -> None:
    m.branches.add(
        {
            "id": bid,
            "from_node": f,
            "to_node": t,
            "resistance": 3.0,
            "reactance": 15.0,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )


def _build_net_with_transit() -> Working:
    """4 узла: slack(gen) — транзит — нагрузка — нагрузка.

    Узел 2 — чистый транзит (exist_load=0, exist_gen=0), через него идёт
    весь переток к узлам 3-4.
    """
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "generation_p": 60.0,
            "generation_q": 20.0,
            "generation_p_min": 0.0,
            "generation_p_max": 200.0,
            "generation_q_min": -100.0,
            "generation_q_max": 100.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 109.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    for nid, vm in ((3, 108.5), (4, 108.0)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": vm,
                "load_p": 30.0,
                "load_q": 10.0,
                "load_p_min": 0.0,
                "load_p_max": 100.0,
                "load_q_min": 0.0,
                "load_q_max": 50.0,
                "exist_load": 1,
                "exist_gen": 0,
                "status": True,
                "node_type": int(NodeType.PQ),
            }
        )
    _build_line(m, 100, 1, 2)
    _build_line(m, 200, 2, 3)
    _build_line(m, 300, 2, 4)

    mid = 1
    for nid, vm in [(1, 110.4), (2, 109.2), (3, 108.6), (4, 108.1)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_VOLTAGE,
                "value": vm,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1
    for nid, p, q in [(3, -30.0, -10.0), (4, -30.0, -10.0)]:
        for kind, val in ((KIND_POWER_INJECTION_P, p), (KIND_POWER_INJECTION_Q, q)):
            m.measurements.add(
                {
                    "id": mid,
                    "object_type": OBJ_NODE,
                    "object_id": nid,
                    "measurement_type": kind,
                    "value": val,
                    "variance": 0.5,
                    "status": True,
                    "quality": 0,
                }
            )
            mid += 1
    return m


def _setup_for(model: Working, **kwargs):
    network_pu = model_to_pu(model)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)
    z = np.zeros(0, dtype=np.float64)
    r = csr_matrix((0, 0), dtype=np.float64)
    mi = MeasurementIndex(
        kind=np.zeros(0, dtype=np.int64),
        object_kind=np.zeros(0, dtype=np.int64),
        object_pos=np.zeros(0, dtype=np.int64),
        branch_side=np.zeros(0, dtype=np.int64),
        meas_id=np.zeros(0, dtype=np.int64),
    )
    return build_ipm_setup(model, network_pu, z, r, mi, layout_base=layout, **kwargs), network_pu


def test_transit_rows_tightened_in_r():
    """С флагом: P- и Q-balance-строки транзита получают заданную σ²,
    остальные узлы (включая slack) — мягкую адаптивную."""
    m = _build_net_with_transit()
    setup, _ = _setup_for(m, transit_balance_sigma2_pu=1e-8)
    sigma2 = setup.r_matrix.diagonal()
    n_balance = 4  # все 4 узла активны
    p_rows = sigma2[0:n_balance]
    q_rows = sigma2[n_balance : 2 * n_balance]
    # Позиция транзитного узла 2 в balance-порядке (порядок node-таблицы).
    transit_i = 1
    assert p_rows[transit_i] == 1e-8
    assert q_rows[transit_i] == 1e-8
    for i in range(n_balance):
        if i == transit_i:
            continue
        assert p_rows[i] > 1e-8
        assert q_rows[i] > 1e-8


def test_transit_rows_soft_by_default():
    """Без флага все balance-строки — одна мягкая σ² (baseline не изменён)."""
    m = _build_net_with_transit()
    setup, _ = _setup_for(m)
    sigma2 = setup.r_matrix.diagonal()
    assert np.unique(sigma2).size == 1


def test_slack_without_exist_flags_not_tightened():
    """Slack без exist_* флагов НЕ пиннится к нулевой инжекции."""
    m = _build_net_with_transit()
    m.nodes.update(1, {"exist_gen": 0})
    setup, _ = _setup_for(m, transit_balance_sigma2_pu=1e-8)
    sigma2 = setup.r_matrix.diagonal()
    n_balance = 4
    # Узел 1 (slack, pos 0 в balance-порядке) остаётся мягким.
    assert sigma2[0] > 1e-8
    assert sigma2[n_balance] > 1e-8


def test_estimate_transit_load_suppressed():
    """End-to-end: с флагом псевдонагрузка на транзите ≈ 0 после reconcile."""
    m_off = _build_net_with_transit()
    estimate(m_off, algorithm="ipm")
    off_load = float(m_off.nodes.get_by_id(2).load_p_estimated)

    m_on = _build_net_with_transit()
    res = estimate(m_on, algorithm="ipm", transit_balance_sigma2_pu=1e-8)
    assert res.success
    on_load = float(m_on.nodes.get_by_id(2).load_p_estimated)
    on_load_q = float(m_on.nodes.get_by_id(2).load_q_estimated)

    # Жёсткий zero-injection давит паразитную псевдонагрузку транзита
    # минимум на порядок относительно мягкого baseline (если она была).
    assert abs(on_load) < 0.05
    assert abs(on_load_q) < 0.05
    if abs(off_load) > 1e-6:
        assert abs(on_load) < abs(off_load)
