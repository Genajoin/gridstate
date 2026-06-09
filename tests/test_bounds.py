"""Единая семантика незаданных границ (gridstate.bounds) + её потребители:
клип WLS-разноса, агрегация лимитов генераторов, box-vars IPM."""

from __future__ import annotations

import math

import numpy as np

from gridstate.bounds import SENTINEL_ABS, is_sentinel, resolve_bounds


# ---------------------------------------------------------------------------
# resolve_bounds / is_sentinel
# ---------------------------------------------------------------------------


def test_resolve_bounds_zero_pair_is_unbounded():
    lo, hi = resolve_bounds(0.0, 0.0)
    assert lo == -math.inf and hi == math.inf


def test_resolve_bounds_sentinel_pair_is_unbounded():
    lo, hi = resolve_bounds(-9999.0, 9999.0)
    assert lo == -math.inf and hi == math.inf


def test_resolve_bounds_half_sentinel_kept_as_is():
    """Полусентинельная пара сохраняется: большая сторона неотличима от
    реальных границ BUS-эквивалентов (±десятки ГВт)."""
    assert resolve_bounds(-9999.0, 120.0) == (-9999.0, 120.0)
    assert resolve_bounds(0.0, 42500.0) == (0.0, 42500.0)


def test_resolve_bounds_inverted_pair_is_unbounded():
    lo, hi = resolve_bounds(50.0, -50.0)
    assert lo == -math.inf and hi == math.inf


def test_resolve_bounds_valid_pair_passthrough():
    assert resolve_bounds(-50.0, 100.0) == (-50.0, 100.0)
    # Полупара с нулём — валидна (не оба нули).
    assert resolve_bounds(0.0, 80.0) == (0.0, 80.0)


def test_is_sentinel():
    assert is_sentinel(9999.0)
    assert is_sentinel(-9999.0)
    assert not is_sentinel(8999.9)
    assert SENTINEL_ABS == 9000.0


# ---------------------------------------------------------------------------
# Клип WLS-разноса: незаданные границы больше не зануляют оценку
# ---------------------------------------------------------------------------


def _model_unbounded_load():
    """load-only узел БЕЗ заданных границ (numpy-default 0,0)."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "LoadNoBounds",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
            # load_*_min/max не задаются — остаются (0, 0).
        }
    )
    return m


def test_inj_split_unset_bounds_do_not_zero_estimate():
    """Раньше пара (0,0) клиповала оценку к нулю — нагрузка терялась."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _model_unbounded_load()
    m.nodes.update(1, {"p_inj_calc": -55.0, "q_inj_calc": -21.0})
    write_node_estimates_from_inj(m)
    n = m.nodes.get_by_id(1)
    assert n.load_p_estimated == 55.0
    assert n.load_q_estimated == 21.0


def test_inj_split_sentinel_bounds_do_not_clip():
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _model_unbounded_load()
    m.nodes.update(
        1,
        {
            "load_p_min": -9999.0,
            "load_p_max": 9999.0,
            "load_q_min": -9999.0,
            "load_q_max": 9999.0,
            "p_inj_calc": -300.0,
            "q_inj_calc": 90.0,
        },
    )
    write_node_estimates_from_inj(m)
    n = m.nodes.get_by_id(1)
    assert n.load_p_estimated == 300.0
    assert n.load_q_estimated == -90.0  # ёмкостная нагрузка не клипуется хламом


def test_inj_split_valid_bounds_still_clip():
    """Валидные границы продолжают клиповать (регрессия поведения)."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _model_unbounded_load()
    m.nodes.update(
        1,
        {
            "load_p_min": 0.0,
            "load_p_max": 80.0,
            "p_inj_calc": -200.0,
            "q_inj_calc": 0.0,
        },
    )
    write_node_estimates_from_inj(m)
    n = m.nodes.get_by_id(1)
    assert n.load_p_estimated == 80.0


# ---------------------------------------------------------------------------
# Агрегация лимитов генераторов: сентинелы не суммируются
# ---------------------------------------------------------------------------


def _nodes_and_gens(gen_rows):
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add({"id": 1, "name": "N1", "voltage_nominal": 110.0, "status": True})
    for g in gen_rows:
        m.generators.add(g)
    return m


def test_aggregate_sentinel_limits_poison_node_pair():
    """Один генератор с сентинельными Q-лимитами → узловая Q-пара = ±9999,
    а не мусорная сумма; P-пара (все валидны) — обычная сумма."""
    from gridstate.telemetry.generators import aggregate_generators_to_node

    m = _nodes_and_gens(
        [
            {
                "id": 11,
                "node_id": 1,
                "status": True,
                "power_output": 40.0,
                "reactive_output": 10.0,
                "power_min": 0.0,
                "power_max": 60.0,
                "reactive_min": -50.0,
                "reactive_max": 100.0,
            },
            {
                "id": 12,
                "node_id": 1,
                "status": True,
                "power_output": 30.0,
                "reactive_output": 5.0,
                "power_min": 10.0,
                "power_max": 40.0,
                "reactive_min": -9999.0,
                "reactive_max": 9999.0,
            },
        ]
    )
    stats = aggregate_generators_to_node(m)
    n = m.nodes.get_by_id(1)
    assert n.generation_p == 70.0
    assert n.generation_q == 15.0
    assert (n.generation_p_min, n.generation_p_max) == (10.0, 100.0)
    # Q-диапазон неизвестен → сентинел, не [-10049, 10099].
    assert (n.generation_q_min, n.generation_q_max) == (-9999.0, 9999.0)
    assert stats["sentinel_q_nodes"] == 1
    assert stats["sentinel_p_nodes"] == 0


def test_aggregate_all_valid_limits_summed():
    from gridstate.telemetry.generators import aggregate_generators_to_node

    m = _nodes_and_gens(
        [
            {
                "id": 11,
                "node_id": 1,
                "status": True,
                "power_output": 40.0,
                "reactive_output": 10.0,
                "power_min": 0.0,
                "power_max": 60.0,
                "reactive_min": -20.0,
                "reactive_max": 30.0,
            },
            {
                "id": 12,
                "node_id": 1,
                "status": True,
                "power_output": 30.0,
                "reactive_output": 5.0,
                "power_min": 10.0,
                "power_max": 40.0,
                "reactive_min": -10.0,
                "reactive_max": 15.0,
            },
        ]
    )
    aggregate_generators_to_node(m)
    n = m.nodes.get_by_id(1)
    assert (n.generation_q_min, n.generation_q_max) == (-30.0, 45.0)


# ---------------------------------------------------------------------------
# IPM box-vars: полное покрытие exist_*-узлов
# ---------------------------------------------------------------------------


def _ipm_setup_for(node_kwargs):
    """build_ipm_setup на минимальной 2-узловой модели (slack + тестовый)."""
    from scipy.sparse import csr_matrix

    from gridstate.constants import NodeType
    from gridstate.preprocessing.ipm_setup import build_ipm_setup
    from gridstate.state import StateLayout
    from gridstate.units import model_to_pu
    from gridstate.working import Working
    from gridstate.z_vector import MeasurementIndex

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "Slack",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "name": "Test",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "status": True,
            "node_type": int(NodeType.PQ),
            **node_kwargs,
        }
    )
    m.branches.add(
        {
            "id": 1,
            "from_node": 1,
            "to_node": 2,
            "resistance": 1.0,
            "reactance": 10.0,
            "status": True,
        }
    )
    network_pu = model_to_pu(m)
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
    return build_ipm_setup(m, network_pu, z, r, mi, layout_base=layout)


def test_ipm_box_created_without_q_bounds():
    """exist_gen-узел БЕЗ Q-границ всё равно получает Qgen box-var
    (раньше — нет, и balance-Q прижимал его реактив к нулю)."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_p": 50.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
            # generation_q_min/max не заданы (0, 0).
        }
    )
    assert setup.layout.pgen_node_pos.size == 1
    assert setup.layout.qgen_node_pos.size == 1  # ключевое: box-var есть
    # Дефолтная коробка ±50 p.u.
    q_idx = setup.layout.pgen_node_pos.size  # qgen-секция после pgen
    assert setup.box_lo[q_idx] == -50.0
    assert setup.box_hi[q_idx] == 50.0


def test_ipm_box_created_for_load_without_bounds():
    setup = _ipm_setup_for({"exist_load": 1, "load_p": 30.0})
    assert setup.layout.pnag_node_pos.size == 1
    assert setup.layout.qnag_node_pos.size == 1


def test_ipm_default_box_no_bus_equiv_prior():
    """Дефолтная широкая коробка НЕ получает tight BUS-эквивалент prior
    (ширина 100 p.u. ≥ порога, но это «нет данных», а не эквивалент)."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_p": 50.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
        }
    )
    # prior-меры в z идут после balance: ни одной prior-строки не должно
    # появиться (prior_sigma2_normal_pu=0 и default-box не BUS-эквив).
    n_balance = 2 * 2  # 2 активных узла × (P+Q)
    assert setup.z.shape[0] == n_balance  # z был пуст; только balance-строки


def test_ipm_real_bus_equiv_still_gets_prior():
    """Реальная широкая коробка из ДАННЫХ по-прежнему BUS-эквивалент."""
    setup = _ipm_setup_for(
        {
            "exist_gen": 1,
            "generation_p": 500.0,
            "generation_p_min": 0.0,
            "generation_p_max": 42500.0,
            "generation_q_min": -1000.0,
            "generation_q_max": 1000.0,
        }
    )
    n_balance = 2 * 2
    # Одна prior-строка на pgen (width 425 p.u. > 100) — bus-equiv tight.
    assert setup.z.shape[0] == n_balance + 1
