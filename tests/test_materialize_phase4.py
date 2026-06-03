"""Ф4.1 (слайс 5, CLASS-2): float-ядро материализации режима над контрактными массивами.

``materialize_injections_from_xml`` расщеплена на двухслойный шов: АДАПТЕР (Блокатор 4)
резолвит наблюдаемый режим из XML-FORMULE (``_xml_observed_injections`` через
``_eval_formula`` над snapshot) → числовой ``obs`` (``{node_id → value}``); ЯДРО
``_materialize_area_on_arrays(nodes_arr, obs, ...)`` делает всю контрактную
float-математику (двухуровневая материализация: вербатим наблюдаемых + area-fill
``k_area·max`` ненаблюдаемых, sentinel-санация, clamp) над ``SE_INPUT.nodes``-массивом,
БЕЗ XML/PSC. Здесь проверяем корректность ядра на голых массивах; бит-в-бит публичного
API — canon transitively + end-to-end дифф OLD-vs-NEW (см. соседний слайс apply_rpn).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT
from gridstate.telemetry.xml_args import _materialize_area_on_arrays


# Поле «нагрузка P» как репрезентативный канал (адаптер вызывает ядро 4× для
# load_p/load_q/generation_p/generation_q с теми же контрактными именами колонок).
LOAD_KW = {
    "max_col": "load_p_max",
    "min_col": "load_p_min",
    "exist_col": "exist_load",
    "set_col": "load_p",
    "max_cap": 5e4,
    "skip_neg_min": -100.0,
    "global_k_fallback": 0.40,
}


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    arr["status"] = True  # default ON; строка может переопределить
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def test_mat_core_verbatim_observed():
    # Наблюдаемый узел → set_col = obs вербатим; прочие нетронуты.
    nodes = _nodes([{"id": 1}, {"id": 2}])
    stats = _materialize_area_on_arrays(nodes, {1: 50.0}, **LOAD_KW)
    assert float(nodes[0]["load_p"]) == 50.0
    assert float(nodes[1]["load_p"]) == 0.0
    assert stats["n_obs"] == 1 and stats["n_fill"] == 0
    assert stats["sum"] == 50.0


def test_mat_core_area_fill_k_from_calibrator():
    # Калибратор района: obs=50, max=100 → k_area=0.5; fill-узел max=200 → 0.5·200=100.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 7, "load_p_max": 200.0, "exist_load": 1},
        ]
    )
    stats = _materialize_area_on_arrays(nodes, {1: 50.0}, **LOAD_KW)
    assert float(nodes[0]["load_p"]) == 50.0
    assert float(nodes[1]["load_p"]) == pytest.approx(100.0)
    assert stats["n_obs"] == 1 and stats["n_fill"] == 1


def test_mat_core_fill_clamped_to_max():
    # k=1.5 (obs=150,max=100) → fill 1.5·200=300 > max=200 → clamp к wmax.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 7, "load_p_max": 200.0, "exist_load": 1},
        ]
    )
    _materialize_area_on_arrays(nodes, {1: 150.0}, **LOAD_KW)
    assert float(nodes[1]["load_p"]) == pytest.approx(200.0)


def test_mat_core_fill_clamped_to_min():
    # k=0.1 → fill 0.1·200=20, но wmin=50 → clamp к wmin (min(max(20,50),200)=50).
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "load_p_min": 0.0, "exist_load": 1},
            {"id": 2, "area_id": 7, "load_p_max": 200.0, "load_p_min": 50.0, "exist_load": 1},
        ]
    )
    _materialize_area_on_arrays(nodes, {1: 10.0}, **LOAD_KW)
    assert float(nodes[1]["load_p"]) == pytest.approx(50.0)


def test_mat_core_fill_skipped_when_max_zero():
    # wmax=0 → не _ok → fill-узел НЕ заполняется.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 7, "load_p_max": 0.0, "exist_load": 1},
        ]
    )
    stats = _materialize_area_on_arrays(nodes, {1: 50.0}, **LOAD_KW)
    assert float(nodes[1]["load_p"]) == 0.0
    assert stats["n_fill"] == 0


def test_mat_core_fill_false_verbatim_only():
    # fill=False → только вербатим наблюдаемых; ненаблюдаемые exist не разносятся.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 7, "load_p_max": 200.0, "exist_load": 1},
        ]
    )
    stats = _materialize_area_on_arrays(nodes, {1: 50.0}, fill=False, **LOAD_KW)
    assert float(nodes[0]["load_p"]) == 50.0
    assert float(nodes[1]["load_p"]) == 0.0
    assert stats["n_fill"] == 0


def test_mat_core_status_off_skipped():
    # Узел OFF не пишется, даже будучи в obs.
    nodes = _nodes(
        [
            {"id": 1, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "status": False, "load_p_max": 200.0, "exist_load": 1},
        ]
    )
    stats = _materialize_area_on_arrays(nodes, {1: 50.0, 2: 99.0}, **LOAD_KW)
    assert float(nodes[1]["load_p"]) == 0.0
    assert stats["n_obs"] == 1


def test_mat_core_global_fallback_for_uncalibrated_area():
    # Район без калибратора → glob = median(all_rat) = 0.5; fill 0.5·200=100.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 9, "load_p_max": 200.0, "exist_load": 1},
        ]
    )
    _materialize_area_on_arrays(nodes, {1: 50.0}, **LOAD_KW)
    assert float(nodes[1]["load_p"]) == pytest.approx(100.0)


def test_mat_core_global_is_flat_median_not_median_of_medians():
    # glob = median ПЛОСКОГО списка всех ratio, НЕ median-of-medians.
    # area1: ratios [0.1,0.2,0.3]; area2: [0.9]. flat-median=0.25 vs med-of-med=0.55.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 1, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 1, "load_p_max": 100.0, "exist_load": 1},
            {"id": 3, "area_id": 1, "load_p_max": 100.0, "exist_load": 1},
            {"id": 4, "area_id": 2, "load_p_max": 100.0, "exist_load": 1},
            {"id": 5, "area_id": 3, "load_p_max": 100.0, "exist_load": 1},  # fill, без калибратора
        ]
    )
    _materialize_area_on_arrays(nodes, {1: 10.0, 2: 20.0, 3: 30.0, 4: 90.0}, **LOAD_KW)
    assert float(nodes[4]["load_p"]) == pytest.approx(25.0)  # 0.25·100 (FLAT)


def test_mat_core_no_exist_not_filled():
    # Ненаблюдаемый без exist_load → не заполняется.
    nodes = _nodes(
        [
            {"id": 1, "area_id": 7, "load_p_max": 100.0, "exist_load": 1},
            {"id": 2, "area_id": 7, "load_p_max": 200.0, "exist_load": 0},
        ]
    )
    stats = _materialize_area_on_arrays(nodes, {1: 50.0}, **LOAD_KW)
    assert float(nodes[1]["load_p"]) == 0.0
    assert stats["n_fill"] == 0


def test_mat_core_sum_rounded_to_one_decimal():
    nodes = _nodes([{"id": 1}, {"id": 2}])
    stats = _materialize_area_on_arrays(nodes, {1: 50.04, 2: 0.04}, **LOAD_KW)
    assert stats["sum"] == 50.1  # round(50.08, 1)


def test_mat_core_gen_negative_min_allowed_with_inf_skip():
    # Ген-путь: skip_neg_min=-1e18 пропускает generation_q_min<0 (норма реактива).
    gen_kw = {
        "max_col": "generation_q_max",
        "min_col": "generation_q_min",
        "exist_col": "exist_gen",
        "set_col": "generation_q",
        "max_cap": 5e4,
        "skip_neg_min": -1e18,
        "global_k_fallback": 0.80,
    }
    nodes = _nodes(
        [
            {
                "id": 1,
                "area_id": 7,
                "generation_q_max": 100.0,
                "generation_q_min": -500.0,
                "exist_gen": 1,
            },
            {
                "id": 2,
                "area_id": 7,
                "generation_q_max": 200.0,
                "generation_q_min": -30.0,
                "exist_gen": 1,
            },
        ]
    )
    # node1 калибратор _ok (wmin=-500 ≥ -1e18) → k=40/100=0.4; node2 fill 0.4·200=80.
    stats = _materialize_area_on_arrays(nodes, {1: 40.0}, fill=True, **gen_kw)
    assert float(nodes[1]["generation_q"]) == pytest.approx(80.0)
    assert stats["n_fill"] == 1
