"""Юнит-тесты ``gridstate.post_processing.refine_anti_overshoot``.

Проверяют логику anti-overshoot уточнения со само-валидацией БЕЗ полного solve:
``resolve``-callback подменяется фейком, который мутирует ``voltage_magnitude``
(имитирует пере-решение). Так детерминированно тестируем три ветки:
  * accept — refine снизил max(V/Vnom) → принят (refined-result, V снижен, добавлены
    tight P/Q-инжекц-pseudo);
  * revert — refine НЕ снизил max(V/Vnom) → откат (base-result, V восстановлен);
  * no-op — нет узла > ceiling → ничего не добавлено, base-result.
"""

from __future__ import annotations

from gridstate.constants import (
    MeasurementObjectType,
    MeasurementType,
    NodeType,
)
from gridstate.post_processing import _max_voltage_ratio, refine_anti_overshoot
from gridstate.working import Working


def _build_model(*, overshoot_pu: float = 1.30):
    """2 узла: slack (real-V, 1.0pu) + PQ-узел с overshoot V и БЕЗ real-V-меры."""
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
            "voltage_magnitude": 110.0 * overshoot_pu,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
            "load_p": 5.0,
            "load_q": 2.0,
            "generation_p": 0.0,
            "generation_q": 0.0,
        }
    )
    # real V-мера ТОЛЬКО на slack — узел 2 «ненаблюдаем по V» (кандидат рычага).
    m.measurements.add(
        {
            "id": 1,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": 1,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": 110.0,
            "variance": 0.1,
            "is_pseudo": False,
            "status": True,
        }
    )
    return m


def _count_pseudo_inj(model) -> int:
    me = model.measurements.to_numpy()
    return int(
        sum(
            1
            for i in range(len(me))
            if bool(me["is_pseudo"][i])
            and int(me["measurement_type"][i])
            in (int(MeasurementType.POWER_INJECTION_P), int(MeasurementType.POWER_INJECTION_Q))
        )
    )


def test_anti_overshoot_accept_lowers_v():
    """resolve снижает V узла 2 → max-V падает → refine принят."""
    m = _build_model(overshoot_pu=1.30)

    def resolve():
        # имитируем эффект tight-инжекции: V садится к физ-уровню 1.05pu
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.05})
        return "REFINED"

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "REFINED"
    assert stats["accepted"] is True
    assert stats["tightened"] == 1
    assert stats["max_ratio_before"] > stats["max_ratio_after"]
    # узел 2 остался сниженным
    assert abs(_max_voltage_ratio(m) - 1.05) < 1e-6
    # добавлены P_inj и Q_inj pseudo на overshoot-узел
    assert _count_pseudo_inj(m) == 2


def test_anti_overshoot_revert_when_not_improved():
    """resolve НЕ снижает V (даже повышает) → max-V не упал → откат к base."""
    m = _build_model(overshoot_pu=1.30)

    def resolve():
        m.nodes.update(2, {"voltage_magnitude": 110.0 * 1.35})  # стало хуже
        return "REFINED"

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"  # откат
    assert stats["accepted"] is False
    # V узла 2 восстановлен к исходному overshoot 1.30 (откат снимка)
    assert abs(_max_voltage_ratio(m) - 1.30) < 1e-6


def test_anti_overshoot_noop_when_no_overshoot():
    """Нет узла > ceiling → resolve не зовётся, ничего не добавлено."""
    m = _build_model(overshoot_pu=1.05)  # в пределах нормы
    calls = {"n": 0}

    def resolve():
        calls["n"] += 1
        return "REFINED"

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"
    assert stats["tightened"] == 0
    assert stats["accepted"] is False
    assert calls["n"] == 0  # resolve не вызывался
    assert _count_pseudo_inj(m) == 0


def test_anti_overshoot_skips_node_with_real_v():
    """Узел с real-V-мерой НЕ трогаем даже при overshoot (наблюдаем — не артефакт)."""
    m = _build_model(overshoot_pu=1.30)
    # дать узлу 2 real-V-меру → исключён из кандидатов
    m.measurements.add(
        {
            "id": 2,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": 2,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": 110.0 * 1.30,
            "variance": 0.1,
            "is_pseudo": False,
            "status": True,
        }
    )
    calls = {"n": 0}

    def resolve():
        calls["n"] += 1
        return "REFINED"

    result, stats = refine_anti_overshoot(m, "BASE", resolve, ceiling=1.15)

    assert result == "BASE"
    assert stats["tightened"] == 0
    assert calls["n"] == 0
