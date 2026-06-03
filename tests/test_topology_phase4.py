"""Ф4.1: решающие ядра топологии работают на контрактных массивах (PSC-free).

Дополняет ``test_topology_port.py`` (тот стережёт бит-в-бит публичного API против
``gridstate.topology``). Здесь проверяем НОВУЮ способность Ф4: каждое ``_*``-ядро

* исполняется на «голых» массивах схемы контракта (``SE_INPUT.*.input_dtype()``),
  БЕЗ ``PowerSystemModel`` — то есть зависит только от контрактных колонок (если бы
  ядро читало неконтрактную колонку, оно бы упало на голом массиве);
* возвращает корректный **план** (id объектов под смену статуса/типа);
* план ядра совпадает с фактическим эффектом публичного адаптера на PSC-модели
  (связка ядро↔адаптер поверх PSC-эквивалентности).
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.topology import (
    _disconnected_nodes_to_disable,
    _gen_nodes_to_promote,
    _isolated_nodes_to_disable,
    _orphan_branches_to_disable,
    _slack_nodes_to_demote,
    disable_disconnected_components,
    disable_isolated_nodes,
    disable_orphan_branches,
    refine_node_types_from_generators,
    refine_slack_to_one,
)


_SLACK, _PQ = 2, 0

# Сценарий-близнец _build_model из test_topology_port (multi-slack, gen→PV, orphan,
# disconnected island {6,7}, isolated {5}), но как «голые» контрактные массивы.
_NODES = [
    # (id, node_type, balance_priority, status)
    (1, _SLACK, 1, True),  # выбранный slack (min bp)
    (2, _SLACK, 2, True),  # демотируется в PQ
    (3, _PQ, 0, True),
    (4, _PQ, 0, True),  # имеет генератор → PV
    (5, _PQ, 0, True),  # изолированный (ветвь 107 off)
    (6, _PQ, 0, True),  # disconnected pair с 7
    (7, _PQ, 0, True),
    (8, _PQ, 0, True),  # в главном острове, источник orphan
    (9, _PQ, 0, False),  # off-узел, цель orphan-ветви
]
_BRANCHES = [
    # (id, from_node, to_node, status)
    (101, 1, 3, True),
    (102, 3, 2, True),
    (103, 3, 4, True),
    (104, 6, 7, True),  # disconnected island
    (105, 8, 3, True),
    (106, 8, 9, True),  # orphan (9 off)
    (107, 5, 3, False),  # off → узел 5 изолирован
]
_GENS = [(1, 4, True)]  # (id, node_id, status)


def _bare_nodes() -> np.ndarray:
    arr = np.zeros(len(_NODES), dtype=SE_INPUT.nodes.input_dtype())
    for i, (nid, ntype, bp, st) in enumerate(_NODES):
        arr[i]["id"] = nid
        arr[i]["node_type"] = ntype
        arr[i]["balance_priority"] = bp
        arr[i]["status"] = st
    return arr


def _bare_branches() -> np.ndarray:
    arr = np.zeros(len(_BRANCHES), dtype=SE_INPUT.branches.input_dtype())
    for i, (bid, fr, to, st) in enumerate(_BRANCHES):
        arr[i]["id"] = bid
        arr[i]["from_node"] = fr
        arr[i]["to_node"] = to
        arr[i]["status"] = st
    return arr


def _bare_gens() -> np.ndarray:
    arr = np.zeros(len(_GENS), dtype=SE_INPUT.generators.input_dtype())
    for i, (gid, nid, st) in enumerate(_GENS):
        arr[i]["id"] = gid
        arr[i]["node_id"] = nid
        arr[i]["status"] = st
    return arr


# ---------------------------------------------------------------------------
# Ядра на голых контрактных массивах (без модели)
# ---------------------------------------------------------------------------


def test_slack_demote_core_on_bare_arrays():
    assert _slack_nodes_to_demote(_bare_nodes(), _bare_branches()) == [2]


def test_gen_promote_core_on_bare_arrays():
    assert _gen_nodes_to_promote(_bare_nodes(), _bare_gens(), None) == [4]


def test_gen_promote_core_vzd_gate_on_bare_arrays():
    # props: узел 4 имеет ген, но vzd=0 → НЕ PV; узел 8 без гена → не трогается.
    props = {4: {"vzd": 0.0, "exist_gen": True}, 8: {"vzd": 1.0, "exist_gen": True}}
    assert _gen_nodes_to_promote(_bare_nodes(), _bare_gens(), props) == []


def test_orphan_branches_core_on_bare_arrays():
    # ветвь 106 (8→9) указывает на off-узел 9.
    assert _orphan_branches_to_disable(_bare_nodes(), _bare_branches()) == [106]


def test_isolated_nodes_core_on_bare_arrays():
    # узел 5: единственная инцидентная ветвь 107 — off.
    assert _isolated_nodes_to_disable(_bare_nodes(), _bare_branches()) == [5]


def test_disconnected_nodes_core_on_bare_arrays():
    # острова {5} (isolated) и {6,7} не имеют active-пути к slack {1,2}.
    plan = _disconnected_nodes_to_disable(_bare_nodes(), _bare_branches())
    assert set(plan) == {5, 6, 7}
    assert len(plan) == 3  # без дублей


def test_disconnected_core_no_slack_returns_empty():
    nodes = _bare_nodes()
    nodes["node_type"] = _PQ  # снять все slack
    assert _disconnected_nodes_to_disable(nodes, _bare_branches()) == []


# ---------------------------------------------------------------------------
# План ядра == эффект адаптера на PSC-модели (связка ядро↔адаптер)
# ---------------------------------------------------------------------------


def _build_psc_model():
    from gridstate.constants import BranchType
    from gridstate.working import Working

    m = Working.empty()
    for nid, ntype, bp, st in _NODES:
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": st,
                "node_type": ntype,
                "balance_priority": bp,
            }
        )
    for bid, fr, to, st in _BRANCHES:
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
    for gid, nid, st in _GENS:
        m.generators.add({"id": gid, "node_id": nid, "status": st, "power_output": 50.0})
    return m


def test_adapter_effect_matches_core_plan():
    """Каждый адаптер меняет ровно те id, что вернуло его ядро (на исходном состоянии)."""
    m = _build_psc_model()

    # slack→PQ
    plan = _slack_nodes_to_demote(m.nodes.to_numpy(), m.branches.to_numpy())
    n = refine_slack_to_one(m)
    assert n == len(plan)
    types = {int(r["id"]): int(r["node_type"]) for r in m.nodes.to_numpy()}
    for nid in plan:
        assert types[nid] == _PQ

    # gen→PV (после refine_slack, как в pipeline)
    plan = _gen_nodes_to_promote(m.nodes.to_numpy(), m.generators.to_numpy(), None)
    n = refine_node_types_from_generators(m)
    assert n == len(plan)
    types = {int(r["id"]): int(r["node_type"]) for r in m.nodes.to_numpy()}
    for nid in plan:
        assert types[nid] == 1  # PV

    # orphan branches
    plan = _orphan_branches_to_disable(m.nodes.to_numpy(), m.branches.to_numpy())
    n = disable_orphan_branches(m)
    assert n == len(plan)
    bstat = {int(r["id"]): bool(r["status"]) for r in m.branches.to_numpy()}
    for bid in plan:
        assert bstat[bid] is False

    # disconnected
    plan = _disconnected_nodes_to_disable(m.nodes.to_numpy(), m.branches.to_numpy())
    n = disable_disconnected_components(m)
    assert n == len(plan)
    nstat = {int(r["id"]): bool(r["status"]) for r in m.nodes.to_numpy()}
    for nid in plan:
        assert nstat[nid] is False

    # isolated
    plan = _isolated_nodes_to_disable(m.nodes.to_numpy(), m.branches.to_numpy())
    n = disable_isolated_nodes(m)
    assert n == len(plan)
