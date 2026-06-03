"""Тесты `normalize_breaker_reactance`.

Загрузчик подменяет ветви-«короткозамыкатели» (R=X=0: секции, выключатели,
блок-связи) на R=0, X=1.0 Ом — ФИКСИРОВАННО в Омах. После model_to_pu это
даёт X_pu, зависящий от класса напряжения: на 10.5 кВ X_pu≈0.9 (катастрофа
для блочных ген-шин), на 500 кВ X_pu≈4e-5 (норма). `normalize_breaker_reactance`
приводит X к volt-aware значению X_pu=eps_pu на любом классе.
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import normalize_breaker_reactance
from gridstate.telemetry.topology import _BREAKER_X_SENTINEL_OHM
from gridstate.units import BASE_MVA


def _toy_model_with_breakers():
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid, vn in ((1, 10.5), (2, 10.5), (3, 500.0), (4, 500.0), (5, 0.0), (6, 0.0), (7, 0.0)):
        m.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": vn,
                "status": True,
                "node_type": int(NodeType.PQ),
            }
        )

    def _br(bid, frm, to, r, x):
        m.branches.add(
            {
                "id": bid,
                "name": f"B{bid}",
                "from_node": frm,
                "to_node": to,
                "resistance": r,
                "reactance": x,
                "parallel_id": 1,
                "tap_ratio": 1.0,
                "status": True,
                "branch_type": int(BranchType.LINE),
            }
        )

    S = _BREAKER_X_SENTINEL_OHM
    _br(100, 1, 2, 0.0, S)  # LV-сентинел → нормализуется
    _br(200, 3, 4, 0.0, S)  # HV-сентинел → нормализуется
    _br(300, 2, 3, 6.05, 30.25)  # реальная ВЛ → НЕ трогаем (R>0)
    _br(400, 5, 6, 0.0, S)  # сентинел, но Vn недоступен → пропуск
    _br(500, 7, 3, 0.0, S)  # from Vn=0 → fallback на to (500 кВ)
    return m


def test_normalize_breaker_reactance_synthetic():
    """Сентинелы R=0,X=1.0 → X_pu=eps_pu на каждом классе; остальное не трогаем."""
    m = _toy_model_with_breakers()
    eps = 1e-3

    stats = normalize_breaker_reactance(m, eps_pu=eps)

    # Нормализованы: B100 (LV), B200 (HV), B500 (fallback на to). B400 — пропуск
    # (нет Vn), B300 — реальная ветвь (R>0).
    assert stats["normalized"] == 3, stats
    assert stats["eps_pu"] == eps

    by_id = {int(b["id"]): b for b in m.branches.to_numpy()}

    # LV-сентинел: X = eps·(10.5²/100); X_pu = X/(Vn²/Sbase) == eps.
    x100 = float(by_id[100]["reactance"])
    assert x100 == pytest.approx(eps * (10.5**2 / BASE_MVA))
    assert x100 / (10.5**2 / BASE_MVA) == pytest.approx(eps)

    # HV-сентинел.
    x200 = float(by_id[200]["reactance"])
    assert x200 == pytest.approx(eps * (500.0**2 / BASE_MVA))
    assert x200 / (500.0**2 / BASE_MVA) == pytest.approx(eps)

    # fallback на to-узел (500 кВ).
    x500 = float(by_id[500]["reactance"])
    assert x500 == pytest.approx(eps * (500.0**2 / BASE_MVA))

    # Реальная ВЛ (R>0) не тронута.
    assert float(by_id[300]["resistance"]) == pytest.approx(6.05)
    assert float(by_id[300]["reactance"]) == pytest.approx(30.25)

    # Сентинел без Vn — оставлен как есть.
    assert float(by_id[400]["reactance"]) == pytest.approx(_BREAKER_X_SENTINEL_OHM)

    # R=0 у нормализованных сохранён.
    assert float(by_id[100]["resistance"]) == 0.0
    assert float(by_id[200]["resistance"]) == 0.0


def test_normalize_breaker_reactance_idempotent():
    """Повторный вызов — no-op (после нормализации X≠1.0 Ом, сентинел не совпадает)."""
    m = _toy_model_with_breakers()
    normalize_breaker_reactance(m)
    stats2 = normalize_breaker_reactance(m)
    assert stats2["normalized"] == 0, stats2


def test_normalize_breaker_reactance_no_branches():
    """Модель без ветвей → 0 нормализаций, без падения."""
    from gridstate.working import Working

    m = Working.empty()
    stats = normalize_breaker_reactance(m)
    assert stats["normalized"] == 0
