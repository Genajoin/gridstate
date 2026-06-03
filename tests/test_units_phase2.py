"""Конвертеры на контракт (``gridstate.units``).

Проверяет, что расщепление конвертеров на **массивные ядра** + тонкие
модель-адаптеры — бит-в-бит (поведение не сместилось):

* read-ядро ``network_pu_from_tables`` ≡ ``model_to_pu`` (поле в поле);
* read-ядро работает на голых контрактных таблицах (без полноценной модели) —
  демонстрация развязки конвертеров от объекта-модели;
* write-ядра ``compute_node_results_pu`` / ``compute_branch_results_pu`` дают
  ровно те же величины, что эталонные формулы pu→именованные единицы.
"""

from __future__ import annotations

import math

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.units import (
    BASE_MVA,
    compute_branch_results_pu,
    compute_node_results_pu,
    model_to_pu,
    network_pu_from_tables,
    write_results_to_model,
)
from gridstate.ybus import build_ybus
from tests.test_units import _build_toy_3bus_model


# --------------------------------------------------------------------------
# read-ядро ≡ model_to_pu (бит-в-бит)
# --------------------------------------------------------------------------

_NETWORK_PU_ARRAY_FIELDS = (
    "bus_ids",
    "bus_vn_kv",
    "bus_type",
    "branch_ids",
    "from_idx",
    "to_idx",
    "branch_r",
    "branch_x",
    "branch_g",
    "branch_b",
    "branch_g_from",
    "branch_b_from",
    "branch_g_to",
    "branch_b_to",
    "tap_ratio",
    "phase_shift",
    "bus_g_shunt",
    "bus_b_shunt",
    "bus_p_injection",
    "bus_q_injection",
)
_NETWORK_PU_SCALAR_FIELDS = ("n_bus", "n_branch", "slack_idx", "base_mva")


def test_network_pu_from_tables_bit_identical_to_model_to_pu():
    m = _build_toy_3bus_model()

    via_model = model_to_pu(m)
    via_tables = network_pu_from_tables(m.nodes.to_numpy(), m.branches.to_numpy())

    for f in _NETWORK_PU_SCALAR_FIELDS:
        assert getattr(via_model, f) == getattr(via_tables, f), f
    for f in _NETWORK_PU_ARRAY_FIELDS:
        a = getattr(via_model, f)
        b = getattr(via_tables, f)
        assert a.dtype == b.dtype, f
        assert np.array_equal(a, b), f


def test_read_core_runs_on_bare_contract_tables_without_model():
    """read-ядро потребляет таблицы входного слоя контракта без полноценной модели."""
    nodes = np.zeros(2, dtype=SE_INPUT.nodes.input_dtype())
    nodes["id"] = [1, 2]
    nodes["voltage_nominal"] = [110.0, 110.0]
    nodes["status"] = [True, True]
    nodes["node_type"] = [2, 0]  # slack, PQ
    nodes["generation_p"] = [40.0, 0.0]
    nodes["load_p"] = [0.0, 30.0]
    nodes["generation_q"] = [10.0, 0.0]
    nodes["load_q"] = [0.0, 12.0]

    branches = np.zeros(1, dtype=SE_INPUT.branches.input_dtype())
    branches["id"] = [10]
    branches["from_node"] = [1]
    branches["to_node"] = [2]
    branches["status"] = [True]
    branches["resistance"] = [12.1]
    branches["reactance"] = [60.5]
    branches["tap_ratio"] = [1.0]
    branches["branch_type"] = [0]

    pu = network_pu_from_tables(nodes, branches)

    assert pu.n_bus == 2
    assert pu.n_branch == 1
    assert pu.slack_idx == 0
    z_base = 110.0**2 / BASE_MVA
    assert math.isclose(pu.branch_r[0], 12.1 / z_base)
    assert math.isclose(pu.branch_x[0], 60.5 / z_base)
    # инъекции = (gen - load) / base_mva
    assert math.isclose(pu.bus_p_injection[1], (0.0 - 30.0) / BASE_MVA)


def test_read_core_multi_slack_picks_min_priority_on_bare_tables():
    """Ветка multi-slack (argmin balance_priority) на голых контрактных массивах."""
    nodes = np.zeros(2, dtype=SE_INPUT.nodes.input_dtype())
    nodes["id"] = [1, 2]
    nodes["voltage_nominal"] = [110.0, 110.0]
    nodes["status"] = [True, True]
    nodes["node_type"] = [2, 2]  # оба slack
    nodes["balance_priority"] = [5, 1]  # первичный — узел 2 (min priority)
    branches = np.zeros(0, dtype=SE_INPUT.branches.input_dtype())

    pu = network_pu_from_tables(nodes, branches)
    assert pu.slack_idx == 1


# --------------------------------------------------------------------------
# write-ядра ≡ эталонные формулы pu→именованные единицы
# --------------------------------------------------------------------------


def _solution(pu):
    """Детерминированное «решение» (не flat) для проверки формул."""
    rng_v = 1.0 + 0.01 * np.arange(pu.n_bus, dtype=np.float64)
    rng_d = 0.02 * np.arange(pu.n_bus, dtype=np.float64)
    return rng_v, rng_d


def test_compute_node_results_pu_matches_reference():
    pu = model_to_pu(_build_toy_3bus_model())
    ybus, _yf, _yt = build_ybus(pu)
    v_pu, delta = _solution(pu)

    voltage_kv, p_inj_mw, q_inj_mvar = compute_node_results_pu(v_pu, delta, pu, ybus=ybus)

    # Эталон: те же формулы, что были инлайн в write_results_to_model.
    ref_kv = v_pu * pu.bus_vn_kv
    v_complex = v_pu * np.exp(1j * delta)
    s_bus = v_complex * np.conj(ybus @ v_complex)
    assert np.array_equal(voltage_kv, ref_kv)
    assert np.array_equal(p_inj_mw, s_bus.real * BASE_MVA)
    assert np.array_equal(q_inj_mvar, s_bus.imag * BASE_MVA)


def test_compute_node_results_pu_without_ybus():
    pu = model_to_pu(_build_toy_3bus_model())
    v_pu, delta = _solution(pu)
    voltage_kv, p_inj_mw, q_inj_mvar = compute_node_results_pu(v_pu, delta, pu, ybus=None)
    assert np.array_equal(voltage_kv, v_pu * pu.bus_vn_kv)
    assert p_inj_mw is None and q_inj_mvar is None


def test_compute_branch_results_pu_matches_reference():
    pu = model_to_pu(_build_toy_3bus_model())
    _ybus, yf, yt = build_ybus(pu)
    v_pu, delta = _solution(pu)

    (p_from, q_from, p_to, q_to, i_from_a, i_to_a, loss_p, loss_q) = compute_branch_results_pu(
        v_pu, delta, pu, yf, yt
    )

    # Эталон: формулы, ранее инлайн в write_results_to_model.
    v_complex = v_pu * np.exp(1j * delta)
    i_from = yf @ v_complex
    i_to = yt @ v_complex
    s_from = v_complex[pu.from_idx] * np.conj(i_from)
    s_to = v_complex[pu.to_idx] * np.conj(i_to)
    sqrt3 = math.sqrt(3.0)
    i_base_from = BASE_MVA * 1000.0 / (sqrt3 * pu.bus_vn_kv[pu.from_idx])
    i_base_to = BASE_MVA * 1000.0 / (sqrt3 * pu.bus_vn_kv[pu.to_idx])

    assert np.array_equal(p_from, s_from.real * BASE_MVA)
    assert np.array_equal(q_from, s_from.imag * BASE_MVA)
    assert np.array_equal(p_to, s_to.real * BASE_MVA)
    assert np.array_equal(q_to, s_to.imag * BASE_MVA)
    assert np.array_equal(i_from_a, np.abs(i_from) * i_base_from)
    assert np.array_equal(i_to_a, np.abs(i_to) * i_base_to)
    assert np.array_equal(loss_p, p_from + p_to)
    assert np.array_equal(loss_q, q_from + q_to)


def test_write_results_to_model_imbalance_path_bit_identical():
    """Прямой путь записи с ybus: p_inj_calc/imbalance через ядро ≡ запись в модель.

    Toy-модель несёт НЕнулевые gen/load (node1 +50, node2 −30, node3 +20 по P) —
    значит вычитание ``p_calc − (gen − load)`` ассертится с ненулевым уменьшаемым,
    исполняя изменённую строку гварда узлового цикла.
    """
    m = _build_toy_3bus_model()
    pu = model_to_pu(m)
    ybus, yf, yt = build_ybus(pu)
    v_pu, delta = _solution(pu)

    # Эталон из ядра (на массивах) — ДО записи в модель.
    voltage_kv, p_inj_mw, q_inj_mvar = compute_node_results_pu(v_pu, delta, pu, ybus=ybus)
    net_p = np.array(
        [
            float(m.nodes.get_by_id(int(nid)).generation_p)
            - float(m.nodes.get_by_id(int(nid)).load_p)
            for nid in pu.bus_ids
        ],
        dtype=np.float64,
    )
    net_q = np.array(
        [
            float(m.nodes.get_by_id(int(nid)).generation_q)
            - float(m.nodes.get_by_id(int(nid)).load_q)
            for nid in pu.bus_ids
        ],
        dtype=np.float64,
    )
    assert np.any(net_p != 0.0)  # путь действительно нетривиален

    write_results_to_model(m, v_pu, delta, pu, yf=yf, yt=yt, ybus=ybus)

    for pos, nid in enumerate(pu.bus_ids.tolist()):
        node = m.nodes.get_by_id(int(nid))
        assert node.voltage_magnitude == float(voltage_kv[pos])
        assert node.p_inj_calc == float(p_inj_mw[pos])
        assert node.q_inj_calc == float(q_inj_mvar[pos])
        assert node.imbalance_p == float(p_inj_mw[pos]) - float(net_p[pos])
        assert node.imbalance_q == float(q_inj_mvar[pos]) - float(net_q[pos])
