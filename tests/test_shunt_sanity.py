"""Юнит-тесты ``gridstate.shunt_sanity`` (research-шаг try-off/flip шунтов).

Классификация на синтетике: кандидат = активный шунт на узле, чья node-V-мера
расходится с решением (|z−h| > v_frac·Vnom). Гейты: нет шунта / малая невязка /
неправдоподобное z — не кандидат; ранжирование по невязке + кап. Правки
``edit_shunt`` (off/flip) и скаляр ``sum_rn2``. Сам шаг зарегистрирован в
contract.run как research-оркестрация полными re-run'ами; default OFF
(бит-в-бит гарантию держит npz-гейт cspase).
"""

from __future__ import annotations

import numpy as np

from gridstate.pipeline import STEPS, PipelineConfig
from gridstate.shunt_sanity import classify_shunt_candidates, edit_shunt, sum_rn2
from gridstate.working import Working
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


def _build(
    *,
    vmag: float = 750.3,
    z: float = 760.3,
    with_shunt: bool = True,
    shunt_status: bool = True,
) -> Working:
    m = Working.empty()
    m.nodes.add(
        {
            "id": 7,
            "voltage_nominal": 750.0,
            "voltage_magnitude": vmag,
            "status": True,
        }
    )
    if with_shunt:
        m.shunts.add(
            {
                "id": 1,
                "node_id": 7,
                "conductance": 0.0,
                "susceptance": -5.33e-4,
                "status": shunt_status,
            }
        )
    m.measurements.add(
        {
            "id": 100,
            "object_type": OBJ_NODE,
            "object_id": 7,
            "measurement_type": KIND_VOLTAGE,
            "value": z,
            "variance": 4.0,
            "status": True,
            "is_pseudo": False,
        }
    )
    return m


def _classify(m: Working, *, v_frac: float = 0.012, max_candidates: int = 6):
    return classify_shunt_candidates(
        m.measurements.to_numpy(),
        m.nodes.to_numpy(),
        m.shunts.to_numpy(),
        v_frac=v_frac,
        max_candidates=max_candidates,
    )


def test_candidate_found():
    # |z−h| = 10 кВ = 1.33% Vnom > 1.2% → кандидат
    plan = _classify(_build())
    assert plan.candidates == (7,)
    assert not plan.empty


def test_no_shunt_not_candidate():
    plan = _classify(_build(with_shunt=False))
    assert plan.empty


def test_inactive_shunt_not_candidate():
    plan = _classify(_build(shunt_status=False))
    assert plan.empty


def test_small_deviation_not_candidate():
    # |z−h| = 5 кВ = 0.67% Vnom < 1.2%
    plan = _classify(_build(z=755.3))
    assert plan.empty


def test_implausible_z_ignored():
    # z вне 0.5–1.5 Vnom (мусор/ноль) — не кандидат
    plan = _classify(_build(z=0.0))
    assert plan.empty


def test_ranking_and_cap():
    m = _build()  # узел 7: невязка 10
    m.nodes.add({"id": 8, "voltage_nominal": 750.0, "voltage_magnitude": 750.0, "status": True})
    m.shunts.add({"id": 2, "node_id": 8, "susceptance": 1e-4, "status": True})
    m.measurements.add(
        {
            "id": 101,
            "object_type": OBJ_NODE,
            "object_id": 8,
            "measurement_type": KIND_VOLTAGE,
            "value": 780.0,  # невязка 30 > 10 → узел 8 первый
            "variance": 4.0,
            "status": True,
            "is_pseudo": False,
        }
    )
    plan = _classify(m)
    assert plan.candidates == (8, 7)
    capped = _classify(m, max_candidates=1)
    assert capped.candidates == (8,)


def test_edit_shunt_off_and_flip():
    m = _build()
    assert edit_shunt(m, 7, "off") == 1
    arr = m.shunts.to_numpy()
    assert not bool(arr["status"][0])
    assert edit_shunt(m, 7, "flip") == 1
    arr = m.shunts.to_numpy()
    assert arr["susceptance"][0] == np.float64(5.33e-4)
    assert edit_shunt(m, 999, "off") == 0  # нет шунтов узла — no-op


def test_sum_rn2_live_real_only():
    m = _build()
    arr = m.measurements.to_numpy()
    arr["estimated_si"][0] = 750.3  # (760.3−750.3)²/4 = 25
    m.measurements.update_from_array(arr)
    assert np.isclose(sum_rn2(m.measurements.to_numpy()), 25.0)
    # выключенная мера не участвует
    arr["status"][0] = False
    m.measurements.update_from_array(arr)
    assert sum_rn2(m.measurements.to_numpy()) == 0.0


def test_toggle_default_off_and_not_a_step():
    # Оркестрация — на уровне contract.run (полные re-run'ы), НЕ шаг пайплайна.
    assert PipelineConfig().shunt_sanity is False
    assert "shunt_sanity" not in [s.name for s in STEPS]
