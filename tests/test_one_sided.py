"""Тесты одностороннего отключения ветвей (gridstate.preprocessing.one_sided)."""

from __future__ import annotations

import math

from gridstate.constants import NodeType
from gridstate.preprocessing.one_sided import (
    FROM_OPEN,
    OFF,
    ON,
    TO_OPEN,
    _driving_point_shunt,
    classify_branch_connectivity,
    fold_one_sided_branches,
)
from gridstate.working import Working


def _model(*, b_line: float = 1e-4, node3_on: bool = False, r: float = 0.01, x: float = 0.1):
    """3 узла, 2 линии: B10=(1-2) обе живые, B11=(1-3) узел 3 управляемый.

    Линия B11 несёт полную зарядную ``b_line`` (См). При ``node3_on=False`` это
    односторонне-отключённая энергизированная линия (живой конец — узел 1).
    """
    m = Working.empty()
    for nid, vn, on in [(1, 500.0, True), (2, 500.0, True), (3, 500.0, node3_on)]:
        m.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": vn,
                "voltage_magnitude": vn,
                "status": on,
                "node_type": int(NodeType.PQ),
                "shunt_b": 0.0,
                "shunt_g": 0.0,
            }
        )
    m.branches.add(
        {
            "id": 10,
            "name": "B10",
            "from_node": 1,
            "to_node": 2,
            "status": True,
            "branch_type": 0,
            "tap_ratio": 1.0,
            "resistance": 0.02,
            "reactance": 0.2,
            "susceptance": 5e-5,
        }
    )
    m.branches.add(
        {
            "id": 11,
            "name": "B11",
            "from_node": 1,
            "to_node": 3,
            "status": True,
            "branch_type": 0,
            "tap_ratio": 1.0,
            "resistance": r,
            "reactance": x,
            "susceptance": b_line,
        }
    )
    return m


def test_classify_one_sided():
    m = _model(node3_on=False)
    cls = classify_branch_connectivity(m)
    assert cls[10] == ON  # обе стороны живы
    assert cls[11] == TO_OPEN  # to=3 отключён, from=1 живой
    # перевернём: пусть отключён from
    m.branches.update(11, {"from_node": 3, "to_node": 1})
    cls = classify_branch_connectivity(m)
    assert cls[11] == FROM_OPEN
    # ветвь off → OFF
    m.branches.update(11, {"status": False})
    assert classify_branch_connectivity(m)[11] == OFF


def test_driving_point_is_full_b_not_half():
    """Чистая зарядная линия: Y_seen ≈ j·b (полная B), НЕ j·b/2."""
    b = 1e-4
    y = _driving_point_shunt(
        r=0.01, x=0.1, g=0.0, b=b, g_live=0.0, b_live=0.0, g_dead=0.0, b_dead=0.0
    )
    # мнимая часть близка к полной B (в пределах ~1%), и заметно больше B/2
    assert math.isclose(y.imag, b, rel_tol=0.02), y
    assert y.imag > 0.9 * b  # точно не half-B
    assert y.real > 0.0  # небольшая активная составляющая через series r


def test_fold_one_sided_adds_full_b_to_live_node():
    m = _model(b_line=1e-4, node3_on=False)
    stats = fold_one_sided_branches(m)
    assert stats["folded"] == 1
    # ветвь погашена
    assert m.branches.get_by_id(11).status is False
    # живой узел 1 получил шунт ≈ полная B
    n1 = m.nodes.get_by_id(1)
    assert math.isclose(n1.shunt_b, 1e-4, rel_tol=0.02), n1.shunt_b
    assert n1.shunt_g > 0.0
    # Q положительна (ёмкостный источник)
    assert float(stats["q_folded_mvar_at_vnom"]) > 0.0
    # мёртвый узел 3 не тронут
    assert m.nodes.get_by_id(3).shunt_b == 0.0


def test_fold_skips_zero_charging_branch():
    """require_charging=True: линия без зарядной B пропускается (ничего не теряем)."""
    m = _model(b_line=0.0, node3_on=False)
    stats = fold_one_sided_branches(m, require_charging=True)
    assert stats["folded"] == 0
    assert stats["skipped_zero_shunt"] == 1
    assert m.branches.get_by_id(11).status is True  # не тронута — достанется orphan-disable


def test_fold_skips_breaker_zero_impedance():
    m = _model(b_line=1e-4, node3_on=False, r=0.0, x=0.0)
    stats = fold_one_sided_branches(m)
    assert stats["folded"] == 0
    assert stats["skipped_breaker"] == 1


def test_fold_noop_when_both_ends_live():
    m = _model(b_line=1e-4, node3_on=True)
    stats = fold_one_sided_branches(m)
    assert stats["folded"] == 0
    assert m.branches.get_by_id(11).status is True
    assert m.nodes.get_by_id(1).shunt_b == 0.0


def test_fold_transformer_skipped_by_default():
    m = _model(b_line=1e-4, node3_on=False)
    m.branches.update(11, {"branch_type": 1, "tap_ratio": 1.05})
    stats = fold_one_sided_branches(m, lines_only=True)
    assert stats["folded"] == 0
    assert stats["skipped_transformer"] == 1
