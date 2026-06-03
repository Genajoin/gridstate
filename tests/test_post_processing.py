"""Постпроцессинг результатов SE: разнос инжекций в load/generation_*_estimated
+ применение статических характеристик нагрузки (синтетические unit-тесты)."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Unit-тесты write_node_estimates_from_inj (без зависимости от XML).
# ---------------------------------------------------------------------------


def _toy_model_for_inj_split():
    """Минимальная модель: 5 узлов с разными exist_load/exist_gen паттернами.

    Узел 1 (slack/gen-only), 2 (load-only), 3 (transit, статус=False —
    проверка что неактивный узел не пишется), 4 (both), 5 (transit).
    """
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "GenOnly",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "generation_p": 80.0,
            "generation_q": 30.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
            "generation_q_min": -50.0,
            "generation_q_max": 50.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "name": "LoadOnly",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "load_p": 40.0,
            "load_q": 15.0,
            "load_p_min": 0.0,
            "load_p_max": 80.0,
            "load_q_min": 0.0,
            "load_q_max": 30.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "name": "TransitOff",
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": False,  # неактивный — функция должна пропустить
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 4,
            "name": "Both",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "generation_p": 60.0,
            "generation_q": 20.0,
            "generation_p_min": 0.0,
            "generation_p_max": 100.0,
            "generation_q_min": -30.0,
            "generation_q_max": 30.0,
            "load_p": 20.0,
            "load_q": 10.0,
            "load_p_min": 0.0,
            "load_p_max": 80.0,
            "load_q_min": 0.0,
            "load_q_max": 30.0,
            "exist_gen": 1,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PV),
        }
    )
    m.nodes.add(
        {
            "id": 5,
            "name": "Transit",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    return m


def _set_inj(model, node_id: int, p: float, q: float) -> None:
    model.nodes.update(node_id, {"p_inj_calc": p, "q_inj_calc": q})


def test_write_node_estimates_from_inj_transit():
    """Узел без exist_load и exist_gen → load/gen_estimated = 0."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 5, 25.0, 10.0)  # ненулевая инжекция на transit
    stats = write_node_estimates_from_inj(m)
    n5 = m.nodes.get_by_id(5)
    assert n5.load_p_estimated == 0.0
    assert n5.load_q_estimated == 0.0
    assert n5.generation_p_estimated == 0.0
    assert n5.generation_q_estimated == 0.0
    assert stats["transit"] == 1  # только узел 5 — статус=True transit


def test_write_node_estimates_from_inj_gen_only():
    """exist_gen=1, exist_load=0 → generation_p_estimated=p_inj, clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 1, 75.0, 25.0)
    write_node_estimates_from_inj(m)
    n1 = m.nodes.get_by_id(1)
    assert n1.generation_p_estimated == 75.0
    assert n1.generation_q_estimated == 25.0
    assert n1.load_p_estimated == 0.0
    assert n1.load_q_estimated == 0.0


def test_write_node_estimates_from_inj_gen_only_clipped():
    """gen_only с p_inj за пределами [gen_p_min, gen_p_max] → clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 1, 150.0, 80.0)  # > gen_p_max=100, > gen_q_max=50
    stats = write_node_estimates_from_inj(m)
    n1 = m.nodes.get_by_id(1)
    assert n1.generation_p_estimated == 100.0  # clipped к gen_p_max
    assert n1.generation_q_estimated == 50.0  # clipped к gen_q_max
    assert stats["clipped"] >= 1


def test_write_node_estimates_from_inj_load_only():
    """exist_load=1, exist_gen=0 → load_p_estimated = -p_inj, clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    # p_inj=-50 (узел потребляет 50 МВт) → load_p_est = 50
    _set_inj(m, 2, -50.0, -20.0)
    write_node_estimates_from_inj(m)
    n2 = m.nodes.get_by_id(2)
    assert n2.load_p_estimated == 50.0
    assert n2.load_q_estimated == 20.0
    assert n2.generation_p_estimated == 0.0
    assert n2.generation_q_estimated == 0.0


def test_write_node_estimates_from_inj_load_only_clipped():
    """load-only с -p_inj за пределами [load_p_min, load_p_max] → clip."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 2, -200.0, -100.0)  # -p_inj=200 > load_p_max=80
    stats = write_node_estimates_from_inj(m)
    n2 = m.nodes.get_by_id(2)
    assert n2.load_p_estimated == 80.0  # clipped к load_p_max
    assert n2.load_q_estimated == 30.0
    assert stats["clipped"] >= 1


def test_write_node_estimates_from_inj_both_proportional():
    """exist_load=1 + exist_gen=1: сохраняется привязка к cur_pg / cur_pn.

    Для узла 4: cur_gen_p=60, cur_load_p=20 → p_inj_meas = 40 МВт.
    Если p_inj_calc = 50 (поток ↑ на 10 МВт): gen покрывает inj+load,
    т.е. gen_est = 50+20 = 70, load_est остаётся = 20.
    """
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 4, 50.0, 25.0)  # gen избыток
    write_node_estimates_from_inj(m)
    n4 = m.nodes.get_by_id(4)
    assert n4.generation_p_estimated == 70.0  # 50 + 20
    assert n4.load_p_estimated == 20.0
    # q_inj=25, cur_q_gen=20, cur_q_load=10 → q_inj_meas = 10
    # q_inj_calc=25 > 0 → gen_q_est = 25 + 10 = 35 → clip к 30 (gen_q_max)
    assert n4.generation_q_estimated == 30.0
    assert n4.load_q_estimated == 10.0


def test_write_node_estimates_from_inj_both_negative_inj():
    """both с отрицательным p_inj (load > gen): load_est растёт."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 4, -30.0, -10.0)  # load избыток
    write_node_estimates_from_inj(m)
    n4 = m.nodes.get_by_id(4)
    # p_inj_calc<0 → gen=cur_gen=60, load = cur_gen - p_inj = 60+30 = 90 → clip к 80
    assert n4.generation_p_estimated == 60.0
    assert n4.load_p_estimated == 80.0  # clipped к load_p_max
    # q_inj<0: gen=cur_gen=20, load = 20+10=30 (=load_q_max, no clip)
    assert n4.generation_q_estimated == 20.0
    assert n4.load_q_estimated == 30.0


def test_write_node_estimates_from_inj_inactive_skipped():
    """Неактивные узлы (status=False) не обновляются."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 3, 999.0, 999.0)
    write_node_estimates_from_inj(m)
    n3 = m.nodes.get_by_id(3)
    # значения по умолчанию (0.0)
    assert n3.load_p_estimated == 0.0
    assert n3.load_q_estimated == 0.0
    assert n3.generation_p_estimated == 0.0
    assert n3.generation_q_estimated == 0.0


# ---------------------------------------------------------------------------
# Unit-тесты apply_load_characteristic (H32, без зависимости от XML).
# ---------------------------------------------------------------------------


def _toy_model_with_sxn():
    """Минимальная модель с СХН для проверки ``apply_load_characteristic``.

    raw_tables['load_models'] — 3 строки:
        idx 0 (sxn_id=1): a0=0.5, a1=0.5, a2=0, b0=1, b1=0, b2=0
        idx 1 (sxn_id=2): a0=1.0, a1=0.0, a2=0 (PQ-const)
        idx 2 (sxn_id=3): a0=0.0, a1=0.0, a2=1.0 (квадратичная)

    Узлы:
        10 — exist_load=1, sxn_id=1, V=Vn → факт. множитель P = 0.5+0.5 = 1.0
        11 — exist_load=1, sxn_id=2, V=0.9·Vn → PQ-const, факт = load_p
        12 — exist_load=1, sxn_id=3, V=1.1·Vn → квадратичный множитель
        13 — exist_load=1, sxn_id=0 (нет СХН) → skipped_no_sxn
        14 — exist_load=0, sxn_id=2 → skipped_no_load
        15 — exist_load=1, sxn_id=99 (за пределами) → skipped_bad_sxn
        16 — status=False, любые поля → пропускается
    """
    from gridstate.constants import NodeType
    from gridstate.contract import SE_INPUT
    from gridstate.working import Working

    m = Working.empty()
    lm = np.zeros(3, dtype=SE_INPUT.raw_table("load_models").numpy_dtype())
    # idx 0 — sxn_id=1: a0=0.5 a1=0.5 a2=0; b: b0=1
    lm[0]["coeff_p_a0"] = 0.5
    lm[0]["coeff_p_a1"] = 0.5
    lm[0]["coeff_p_a2"] = 0.0
    lm[0]["coeff_q_b0"] = 1.0
    lm[0]["coeff_q_b1"] = 0.0
    lm[0]["coeff_q_b2"] = 0.0
    # idx 1 — sxn_id=2: PQ-const
    lm[1]["coeff_p_a0"] = 1.0
    lm[1]["coeff_q_b0"] = 1.0
    # idx 2 — sxn_id=3: чисто квадратичная
    lm[2]["coeff_p_a2"] = 1.0
    lm[2]["coeff_q_b2"] = 1.0
    m.raw_tables["load_models"] = lm

    for nid, kwargs in [
        (
            10,
            {
                "sxn_id": 1,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,  # V_pu=1.0
                "load_p": 100.0,
                "load_q": 50.0,
                "exist_load": 1,
            },
        ),
        (
            11,
            {
                "sxn_id": 2,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 99.0,  # V_pu=0.9
                "load_p": 80.0,
                "load_q": 40.0,
                "exist_load": 1,
            },
        ),
        (
            12,
            {
                "sxn_id": 3,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 121.0,  # V_pu=1.1
                "load_p": 200.0,
                "load_q": 100.0,
                "exist_load": 1,
            },
        ),
        (
            13,
            {
                "sxn_id": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "load_p": 50.0,
                "load_q": 25.0,
                "exist_load": 1,
            },
        ),
        (
            14,
            {
                "sxn_id": 2,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "exist_load": 0,
            },
        ),
        (
            15,
            {
                "sxn_id": 99,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "load_p": 10.0,
                "load_q": 5.0,
                "exist_load": 1,
            },
        ),
        (
            16,
            {
                "sxn_id": 1,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "load_p": 999.0,
                "load_q": 999.0,
                "exist_load": 1,
                "status": False,
            },
        ),
    ]:
        defaults = {
            "id": nid,
            "name": f"N{nid}",
            "status": True,
            "node_type": int(NodeType.PQ),
        }
        defaults.update(kwargs)
        m.nodes.add(defaults)
    return m


def test_apply_load_characteristic_linear():
    """sxn_id=1 (a0=0.5, a1=0.5) при V_pu=1.0 → factor=1.0 → load_p_est = load_p_nom."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    stats = apply_load_characteristic(m)
    n10 = m.nodes.get_by_id(10)
    assert abs(n10.load_p_estimated - 100.0) < 1e-9, n10.load_p_estimated
    # Q: b0=1, b1=b2=0 → factor_q=1.0 → load_q_est = 50.0
    assert abs(n10.load_q_estimated - 50.0) < 1e-9, n10.load_q_estimated
    assert stats["updated"] >= 1


def test_apply_load_characteristic_pq_const():
    """sxn_id=2 (PQ-const): load_*_est = load_*_nom при любом V_pu."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    apply_load_characteristic(m)
    n11 = m.nodes.get_by_id(11)
    # V_pu=0.9 не влияет: factor=1.0
    assert abs(n11.load_p_estimated - 80.0) < 1e-9
    assert abs(n11.load_q_estimated - 40.0) < 1e-9


def test_apply_load_characteristic_quadratic():
    """sxn_id=3 (a2=1, b2=1): factor = V_pu²."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    apply_load_characteristic(m)
    n12 = m.nodes.get_by_id(12)
    # V_pu=1.1 → factor = 1.21
    assert abs(n12.load_p_estimated - 200.0 * 1.21) < 1e-9
    assert abs(n12.load_q_estimated - 100.0 * 1.21) < 1e-9


def test_apply_load_characteristic_counters():
    """skipped_no_sxn / skipped_no_load / skipped_bad_sxn должны быть на месте."""
    from gridstate.post_processing import apply_load_characteristic

    m = _toy_model_with_sxn()
    stats = apply_load_characteristic(m)
    # 3 узла обновлено (10, 11, 12); 13 sxn=0; 14 exist_load=0; 15 bad sxn; 16 status=False
    assert stats["updated"] == 3
    assert stats["skipped_no_sxn"] == 1
    assert stats["skipped_no_load"] == 1
    assert stats["skipped_bad_sxn"] == 1


def test_apply_load_characteristic_no_load_models():
    """Если в модели нет load_models — функция возвращает no_load_models=1."""
    from gridstate.constants import NodeType
    from gridstate.post_processing import apply_load_characteristic
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "load_p": 100.0,
            "exist_load": 1,
            "sxn_id": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    stats = apply_load_characteristic(m)
    assert stats["no_load_models"] == 1
    assert stats["updated"] == 0


def test_write_node_estimates_from_inj_stats_counters():
    """Счётчики по категориям совпадают с реальной разбивкой."""
    from gridstate.post_processing import write_node_estimates_from_inj

    m = _toy_model_for_inj_split()
    _set_inj(m, 1, 50.0, 20.0)
    _set_inj(m, 2, -30.0, -10.0)
    _set_inj(m, 4, 10.0, 5.0)
    stats = write_node_estimates_from_inj(m)
    # 4 активных узла: 1=gen_only, 2=load_only, 4=both, 5=transit; 3=off
    assert stats["gen_only"] == 1
    assert stats["load_only"] == 1
    assert stats["both"] == 1
    assert stats["transit"] == 1
    assert stats["updated"] == 4
