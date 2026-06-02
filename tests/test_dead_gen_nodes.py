"""Тесты каскада мёртвых ген-узлов (gridstate.preprocessing.dead_gen_nodes).

Проверяют КОНТРАКТ правила (гасит ген-стаб с выключенными генами без нагрузки,
degree-1), а НЕ производственную целесообразность (она net-neutral, шаг паркован).
"""

from __future__ import annotations

from gridstate.constants import NodeType
from gridstate.preprocessing.dead_gen_nodes import disable_dead_generator_nodes
from gridstate.working import Working


def _model(*, gen_on: bool = False, load: float = 0.0, extra_branch: bool = False):
    """Сеть: slack(1) — повышающая ветвь — ген-узел(2, генератор). Опц. транзит 2-3."""
    m = Working.empty()
    for nid, ntype in [(1, NodeType.SLACK), (2, NodeType.PQ), (3, NodeType.PQ)]:
        m.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "status": True,
                "node_type": int(ntype),
                "load_p": load if nid == 2 else 0.0,
                "load_q": 0.0,
            }
        )
    m.branches.add(
        {
            "id": 12,
            "name": "step-up",
            "from_node": 1,
            "to_node": 2,
            "status": True,
            "branch_type": 1,
            "tap_ratio": 1.0,
            "resistance": 0.5,
            "reactance": 5.0,
        }
    )
    if extra_branch:
        m.branches.add(
            {
                "id": 23,
                "name": "transit",
                "from_node": 2,
                "to_node": 3,
                "status": True,
                "branch_type": 0,
                "tap_ratio": 1.0,
                "resistance": 0.5,
                "reactance": 5.0,
            }
        )
    m.generators.add({"id": 100, "name": "G1", "node_id": 2, "status": gen_on})
    return m


def test_disables_dead_gen_stub():
    m = _model(gen_on=False)
    res = disable_dead_generator_nodes(m)
    assert res["disabled_nodes"] == 1
    assert res["node_ids"] == [2]
    assert m.nodes.get_by_id(2).status is False


def test_keeps_node_with_live_generator():
    m = _model(gen_on=True)
    res = disable_dead_generator_nodes(m)
    assert res["disabled_nodes"] == 0
    assert m.nodes.get_by_id(2).status is True


def test_keeps_node_with_load():
    m = _model(gen_on=False, load=50.0)
    assert disable_dead_generator_nodes(m)["disabled_nodes"] == 0
    assert m.nodes.get_by_id(2).status is True


def test_keeps_transit_node_above_max_degree():
    """degree-2 ген-узел (несёт транзит) — не трогаем при max_degree=1."""
    m = _model(gen_on=False, extra_branch=True)
    assert disable_dead_generator_nodes(m, max_degree=1)["disabled_nodes"] == 0
    assert m.nodes.get_by_id(2).status is True


def test_never_disables_slack():
    m = _model(gen_on=False)
    # сделаем ген-узел slack
    m.nodes.update(2, {"node_type": int(NodeType.SLACK)})
    assert disable_dead_generator_nodes(m)["disabled_nodes"] == 0


def test_node_without_generators_untouched():
    m = _model(gen_on=False)
    # уберём генератор → узел 2 без генераторов, не ген-стаб
    m.generators.update(100, {"status": True})  # неважно
    m2 = Working.empty()
    for nid, ntype in [(1, NodeType.SLACK), (2, NodeType.PQ)]:
        m2.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "status": True,
                "node_type": int(ntype),
                "load_p": 0.0,
                "load_q": 0.0,
            }
        )
    m2.branches.add(
        {
            "id": 12,
            "name": "b",
            "from_node": 1,
            "to_node": 2,
            "status": True,
            "branch_type": 0,
            "tap_ratio": 1.0,
            "resistance": 0.5,
            "reactance": 5.0,
        }
    )
    assert disable_dead_generator_nodes(m2)["disabled_nodes"] == 0
