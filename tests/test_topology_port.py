"""Топология на gridstate-контейнере.

`gridstate.topology` перенесена из исходной реализации дословно. Этот тест —
функциональный страж: прогон топо-функций на gridstate-модели даёт ожидаемые
статусы/типы узлов и статусы ветвей + ожидаемые возвраты. Дёшево (без
SE-решения) → годится как per-phase гейт при отложенном canon
(см. feedback-defer-heavy-gate-migration).
"""

from __future__ import annotations

import gridstate.topology as cs


def _build_model():
    """Модель, триггерящая все 5 топо-функций (multi-slack, gen→PV, orphan,
    disconnected island, isolated node)."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    nodes = [
        (1, NodeType.SLACK, 1, True),  # выбранный slack (min bp)
        (2, NodeType.SLACK, 2, True),  # демотируется в PQ
        (3, NodeType.PQ, 0, True),
        (4, NodeType.PQ, 0, True),  # имеет генератор → PV
        (5, NodeType.PQ, 0, True),  # изолированный (ветвь b7 off)
        (6, NodeType.PQ, 0, True),  # disconnected pair с 7
        (7, NodeType.PQ, 0, True),
        (8, NodeType.PQ, 0, True),  # в главном острове, источник orphan
        (9, NodeType.PQ, 0, False),  # off-узел, цель orphan-ветви
    ]
    for nid, ntype, bp, st in nodes:
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": st,
                "node_type": int(ntype),
                "balance_priority": bp,
            }
        )
    branches = [
        (101, 1, 3, True),
        (102, 3, 2, True),
        (103, 3, 4, True),
        (104, 6, 7, True),  # disconnected island
        (105, 8, 3, True),
        (106, 8, 9, True),  # orphan (9 off)
        (107, 5, 3, False),  # off → узел 5 изолирован
    ]
    for bid, fr, to, st in branches:
        m.branches.add(
            {
                "id": bid,
                "from_node": fr,
                "to_node": to,
                "resistance": 6.05,
                "reactance": 30.25,
                "status": st,
                "branch_type": int(BranchType.LINE),
            }
        )
    m.generators.add({"id": 1, "node_id": 4, "status": True, "power_output": 50.0})
    return m


def _node_state(m) -> dict:
    return {int(r["id"]): (bool(r["status"]), int(r["node_type"])) for r in m.nodes.to_numpy()}


def _branch_state(m) -> dict:
    return {int(r["id"]): bool(r["status"]) for r in m.branches.to_numpy()}


def test_topology_sequence_runs_clean():
    from gridstate.constants import NodeType

    a = _build_model()

    steps = [
        "refine_slack_to_one",
        "refine_node_types_from_generators",
        "disable_orphan_branches",
        "disable_disconnected_components",
        "disable_isolated_nodes",
    ]
    for name in steps:
        ra = getattr(cs, name)(a)
        assert isinstance(ra, int), f"{name}: возврат не int ({ra!r})"

    node_state = _node_state(a)
    branch_state = _branch_state(a)

    # multi-slack: узел 1 остаётся SLACK (min bp), узел 2 демотируется в PQ
    assert node_state[1][1] == int(NodeType.SLACK)
    assert node_state[2][1] == int(NodeType.PQ)
    # узел 4 имеет ген → PV
    assert node_state[4][1] == int(NodeType.PV)
    # orphan-ветвь 106 (9 off) отключена
    assert branch_state[106] is False
    # disconnected island 6-7 отключена от главного острова (узлы гасятся)
    assert node_state[6][0] is False
    assert node_state[7][0] is False
    # изолированный узел 5 (ветвь 107 off) отключён
    assert node_state[5][0] is False


def test_refine_node_types_with_props_vzd_gate():
    # Ветка node_load_props (vzd-гейт): узел 4 имеет ген, но vzd=0 → НЕ PV;
    # узел 8 в props с vzd>0 но без гена → не трогается.
    from gridstate.constants import NodeType

    a = _build_model()
    cs.refine_slack_to_one(a)
    props = {4: {"vzd": 0.0, "exist_gen": True}, 8: {"vzd": 1.0, "exist_gen": True}}
    ra = cs.refine_node_types_from_generators(a, node_load_props=props)
    assert ra == 0  # узел 4 имеет ген, но vzd=0 → не PV; 8 без гена
    node_state = _node_state(a)
    assert node_state[4][1] != int(NodeType.PV)


def test_no_slack_disconnected_noop():
    # Без slack disable_disconnected_components → 0.
    from gridstate.working import Working

    def _no_slack():
        m = Working.empty()
        m.nodes.add({"id": 1, "voltage_nominal": 110.0, "status": True, "node_type": 0})
        m.nodes.add({"id": 2, "voltage_nominal": 110.0, "status": True, "node_type": 0})
        m.branches.add(
            {
                "id": 1,
                "from_node": 1,
                "to_node": 2,
                "resistance": 1.0,
                "reactance": 10.0,
                "status": True,
                "branch_type": 0,
            }
        )
        return m

    a = _no_slack()
    assert cs.disable_disconnected_components(a) == 0
    node_state = _node_state(a)
    assert node_state[1][0] is True
    assert node_state[2][0] is True
