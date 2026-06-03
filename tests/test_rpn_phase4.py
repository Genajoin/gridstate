"""Ф4.1 (слайс 4, CLASS-2): float-ядро apply_rpn над контрактными массивами.

``apply_rpn_from_xml`` расщеплена на двухслойный шов: АДАПТЕР (Блокатор 4) резолвит
№ отпаек через ``_eval_rpn_arg`` (FormulaEvaluator над snapshot) → ``resolved_taps``;
ЯДРО ``_apply_rpn_on_arrays(branches_arr, shema_ktr_arr, resolved_taps)`` делает всю
контрактную float-математику (ktr-lookup, main-vs-vc выбор, hypot/atan2 tap+phase,
shunt-rescale, diff-stats) над ``SE_INPUT``-массивами, БЕЗ XML/PSC. Здесь проверяем
корректность ядра на голых массивах; бит-в-бит публичного API — canon transitively +
end-to-end дифф OLD-vs-NEW.
"""

from __future__ import annotations

import math

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.telemetry.rpn import _apply_rpn_on_arrays


def _branches(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.branches.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _shema_ktr(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.raw_table("shema_ktr").numpy_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def test_rpn_core_basic_tap_phase_and_shunt():
    # Одна trafo-ветвь, одна строка SHEMA_KTR (main-пара) → tap=1/hypot, shunt·factor².
    branches = _branches(
        [
            {
                "id": 100,
                "tap_ratio": 1.05,
                "phase_shift": 0.0,
                "conductance": 10.0,
                "susceptance": 20.0,
            }
        ]
    )
    sk = _shema_ktr([{"type_rpn": 996, "num_a": 5, "num_r": 0, "ktr_a": 0.95, "ktr_r": 0.0}])
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 0)])
    assert stats["applied"] == 1
    assert stats["shunt_recalc"] == 1
    new_tap = float(branches[0]["tap_ratio"])
    assert new_tap == 1.0 / 0.95  # hypot(1/0.95, 0) = 1/0.95 (im=0)
    assert float(branches[0]["phase_shift"]) == 0.0
    factor = (new_tap / 1.05) ** 2
    assert float(branches[0]["conductance"]) == 10.0 * factor
    assert float(branches[0]["susceptance"]) == 20.0 * factor


def test_rpn_core_num_pbv_selection_closest_to_xml_tap():
    # Две строки одного ключа (NUM_PBV-дубли) → выбираем ту, чей tap ближе к xml_tap=1.05.
    branches = _branches([{"id": 100, "tap_ratio": 1.05}])
    sk = _shema_ktr(
        [
            {"type_rpn": 996, "num_a": 5, "num_r": 0, "ktr_a": 0.90},  # tap=1.111, далёкий
            {"type_rpn": 996, "num_a": 5, "num_r": 0, "ktr_a": 0.95},  # tap=1.0526, ближний
        ]
    )
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 0)])
    assert stats["applied"] == 1
    assert float(branches[0]["tap_ratio"]) == 1.0 / 0.95  # выбран ближний кандидат


def test_rpn_core_main_vs_vc_picks_closest():
    # Строка с обеими парами: main(0.90→1.111) vs vc(0.95→1.0526); xml_tap=1.05 → vc.
    branches = _branches([{"id": 100, "tap_ratio": 1.05}])
    sk = _shema_ktr(
        [
            {
                "type_rpn": 996,
                "num_a": 5,
                "num_r": 0,
                "ktr_a": 0.90,
                "ktr_r": 0.0,
                "ktr_a_vc": 0.95,
                "ktr_r_vc": 0.0,
            }
        ]
    )
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 0)])
    assert stats["applied"] == 1
    assert float(branches[0]["tap_ratio"]) == 1.0 / 0.95  # vc ближе к xml_tap


def test_rpn_core_skipped_empty_ktr_leaves_branch():
    # Все коэффициенты строки нулевые → skipped_empty_ktr, ветвь НЕ трогается.
    branches = _branches([{"id": 100, "tap_ratio": 1.05, "conductance": 10.0}])
    sk = _shema_ktr([{"type_rpn": 996, "num_a": 5, "num_r": 0}])  # all-zero
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 0)])
    assert stats["applied"] == 0
    assert stats["skipped_empty_ktr"] == 1
    assert float(branches[0]["tap_ratio"]) == 1.05  # не тронут
    assert float(branches[0]["conductance"]) == 10.0


def test_rpn_core_skipped_no_ktr_when_key_absent():
    branches = _branches([{"id": 100, "tap_ratio": 1.05}])
    sk = _shema_ktr([{"type_rpn": 996, "num_a": 7, "num_r": 0, "ktr_a": 0.95}])  # другой num_a
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 0)])
    assert stats["applied"] == 0
    assert stats["skipped_no_ktr"] == 1
    assert float(branches[0]["tap_ratio"]) == 1.05


def test_rpn_core_num_r_fallback_to_zero():
    # num_r=3 нет в lookup, но есть (type,num_a,0) → fallback на num_r=0.
    branches = _branches([{"id": 100, "tap_ratio": 1.05}])
    sk = _shema_ktr([{"type_rpn": 996, "num_a": 5, "num_r": 0, "ktr_a": 0.95}])
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 3)])
    assert stats["applied"] == 1
    assert float(branches[0]["tap_ratio"]) == 1.0 / 0.95


def test_rpn_core_skip_counters_passthrough():
    # Адаптерные счётчики (skipped_no_branch/skipped_no_tm) проходят в stats.
    branches = _branches([{"id": 100, "tap_ratio": 1.05}])
    sk = _shema_ktr([{"type_rpn": 996, "num_a": 5, "num_r": 0, "ktr_a": 0.95}])
    stats = _apply_rpn_on_arrays(
        branches, sk, [(100, 996, 5, 0)], skipped_no_branch=4, skipped_no_tm=7
    )
    assert stats["skipped_no_branch"] == 4
    assert stats["skipped_no_tm"] == 7
    assert stats["applied"] == 1


def test_rpn_core_phase_shift_from_ktr_r():
    # Поперечный РПН (ktr_r≠0) → phase_shift = atan2(im, re), im=-ktr_r/z2.
    branches = _branches([{"id": 100, "tap_ratio": 1.0}])
    sk = _shema_ktr([{"type_rpn": 996, "num_a": 5, "num_r": 2, "ktr_a": 1.0, "ktr_r": 0.1}])
    stats = _apply_rpn_on_arrays(branches, sk, [(100, 996, 5, 2)])
    assert stats["applied"] == 1
    z2 = 1.0 * 1.0 + 0.1 * 0.1
    re = 1.0 / z2
    im = -0.1 / z2
    assert float(branches[0]["tap_ratio"]) == math.hypot(re, im)
    assert float(branches[0]["phase_shift"]) == math.atan2(im, re)


def test_rpn_core_empty_shema_ktr_all_skipped():
    branches = _branches([{"id": 100, "tap_ratio": 1.05}])
    stats = _apply_rpn_on_arrays(branches, None, [(100, 996, 5, 0)])
    assert stats["applied"] == 0
    assert stats["skipped_no_ktr"] == 1
