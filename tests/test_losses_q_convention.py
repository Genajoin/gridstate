"""Тест знаковой конвенции Q-потерь в ``compute_system_losses``.

Отчёт о потерях выражает Q-шунты в ФИЗИЧЕСКОЙ конвенции (= dq эталонной SE):
ёмкостный шунт (БК / зарядная B ВЛ) ВЫДАЁТ Q → отрицательные потери;
индуктивный (ШР) ПОГЛОЩАЕТ → положительные. Наша storage-susceptance
инвертирована (конвенция входного формата: b<0 ёмкостный, b>0 индуктивный),
поэтому Q-шунт отрицается (``_SHUNT_Q_SIGN``). См. docstring losses.py.
"""

from __future__ import annotations


def _node_model(shunt_b: float):
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    # SLACK — model_to_pu требует ровно один slack-узел.
    m.nodes.add(
        {
            "id": 1,
            "name": "N1",
            "voltage_nominal": 500.0,
            "voltage_magnitude": 500.0,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
            "shunt_g": 0.0,
            "shunt_b": shunt_b,
        }
    )
    return m


def test_inductive_shunt_shr_absorbs_positive():
    """ШР (b>0 в storage входного формата) → ПОГЛОЩЕНИЕ Q → q_node_shunt > 0."""
    from gridstate.losses import compute_system_losses

    r = compute_system_losses(_node_model(50.0))
    assert r.q_node_shunt > 0, f"ШР должен поглощать (>0), получено {r.q_node_shunt}"


def test_capacitive_shunt_bk_generates_negative():
    """БК / зарядная B (b<0 в storage) → ВЫДАЧА Q → q_node_shunt < 0 (как эталонная SE)."""
    from gridstate.losses import compute_system_losses

    r = compute_system_losses(_node_model(-50.0))
    assert r.q_node_shunt < 0, f"БК должен выдавать (<0), получено {r.q_node_shunt}"


def test_shunt_q_sign_symmetric():
    """Знак симметричен: q_node_shunt(+b) == -q_node_shunt(-b)."""
    from gridstate.losses import compute_system_losses

    pos = compute_system_losses(_node_model(50.0)).q_node_shunt
    neg = compute_system_losses(_node_model(-50.0)).q_node_shunt
    assert abs(pos + neg) < 1e-6 * max(abs(pos), 1.0)
