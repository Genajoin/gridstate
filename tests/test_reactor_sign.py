"""Тесты ``apply_reactors_to_node_shunt`` — знак при сложении B/G (kwarg ``sign``).

Конвенция эталонной SE ``reactors.susceptance`` обратна EE Y-bus: эталонная SE ``B>0``=ШР
(индуктивный), ``B<0``=БК (ёмкостный); в EE ``shunt_b>0``=ёмкостный. Физически
корректный по Q-балансу вариант — ``sign=-1`` (флипает конвенцию). Default
``sign=1`` сохранён (историческое поведение), т.к. ``sign=-1`` как production-
default регрессирует V/δ (см. memory reactor_sign_q_balance_finding).
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import apply_reactors_to_node_shunt


def _model_with_reactor(
    susceptance_uS: float,
    *,
    conductance_uS: float = 0.0,
    reac_status: bool = True,
    node_status: bool = True,
):
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "N1",
            "voltage_nominal": 500.0,
            "status": node_status,
            "node_type": int(NodeType.PQ),
            "shunt_b": 0.0,
            "shunt_g": 0.0,
        }
    )
    m.raw_tables["reactors"] = [
        {
            "id": 1,
            "name": "R1",
            "node_id": 1,
            "num": 1,
            "reac_id_rastr": 0,
            "conductance": conductance_uS,
            "susceptance": susceptance_uS,
            "status": reac_status,
            "ems": 0,
        }
    ]
    return m


def test_sign_default_is_positive_noop():
    """Default (без kwarg) == sign=1 == историческое `shunt_b += B·1e-6`."""
    m_def = _model_with_reactor(100_000.0)
    m_pos = _model_with_reactor(100_000.0)

    s_def = apply_reactors_to_node_shunt(m_def)
    s_pos = apply_reactors_to_node_shunt(m_pos, sign=1)

    assert s_def["applied"] == 1
    assert m_def.nodes.get_by_id(1).shunt_b == pytest.approx(0.1)  # +100000·1e-6
    # default идентичен sign=1
    assert m_def.nodes.get_by_id(1).shunt_b == pytest.approx(m_pos.nodes.get_by_id(1).shunt_b)
    assert s_def["sum_b_added_S"] == pytest.approx(s_pos["sum_b_added_S"])


def test_sign_negative_flips_shr():
    """ШР (B>0): sign=-1 → shunt_b = −B·1e-6 (индуктивный в EE)."""
    m = _model_with_reactor(100_000.0)
    stats = apply_reactors_to_node_shunt(m, sign=-1)
    assert stats["applied"] == 1
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(-0.1)
    assert stats["sum_b_added_S"] == pytest.approx(-0.1)


def test_sign_symmetric_for_bk():
    """БК (B<0): sign=1 → −0.1; sign=-1 → +0.1 (симметрично)."""
    m_pos = _model_with_reactor(-100_000.0)
    m_neg = _model_with_reactor(-100_000.0)
    apply_reactors_to_node_shunt(m_pos, sign=1)
    apply_reactors_to_node_shunt(m_neg, sign=-1)
    assert m_pos.nodes.get_by_id(1).shunt_b == pytest.approx(-0.1)
    assert m_neg.nodes.get_by_id(1).shunt_b == pytest.approx(0.1)


def test_sign_applies_to_conductance_too():
    """sign флипает и G (shunt_g), не только B."""
    m = _model_with_reactor(0.0, conductance_uS=50_000.0)
    apply_reactors_to_node_shunt(m, sign=-1)
    assert m.nodes.get_by_id(1).shunt_g == pytest.approx(-0.05)


def test_inactive_reactor_skipped():
    """status=False реактор не складывается ни при каком sign."""
    m = _model_with_reactor(100_000.0, reac_status=False)
    stats = apply_reactors_to_node_shunt(m, sign=-1)
    assert stats["applied"] == 0
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)


def test_inactive_node_skipped():
    """Реактор на выключенном узле не складывается."""
    m = _model_with_reactor(100_000.0, node_status=False)
    stats = apply_reactors_to_node_shunt(m, sign=-1)
    assert stats["applied"] == 0
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)
