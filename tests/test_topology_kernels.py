"""The topology decision kernels work on contract arrays (vendor-free).

Complements ``test_topology_port.py``, which covers the public functions of
``gridstate.topology``. Here we check the new capability: each ``_*`` kernel

* runs on "bare" arrays of the contract schema (``SE_INPUT.*.input_dtype()``),
  without a full model, i.e. it depends only on contract columns (a kernel that
  read a non-contract column would fail on a bare array);
* returns a correct **plan** (ids of objects whose status/type changes);
* the kernel plan matches the actual effect of the public adapter on a model
  (kernel <-> adapter link).
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.topology import (
    _disconnected_nodes_to_disable,
    _gen_nodes_to_demote,
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

# Twin scenario of _build_model from test_topology_port (multi-slack, gen->PV, orphan,
# disconnected island {6,7}, isolated {5}), but as "bare" contract arrays.
_NODES = [
    # (id, node_type, balance_priority, status)
    (1, _SLACK, 1, True),  # selected slack (min bp)
    (2, _SLACK, 2, True),  # demoted to PQ
    (3, _PQ, 0, True),
    (4, _PQ, 0, True),  # has a generator -> PV
    (5, _PQ, 0, True),  # isolated (branch 107 off)
    (6, _PQ, 0, True),  # disconnected pair with 7
    (7, _PQ, 0, True),
    (8, _PQ, 0, True),  # in the main island, source of the orphan branch
    (9, _PQ, 0, False),  # off node, target of the orphan branch
]
_BRANCHES = [
    # (id, from_node, to_node, status)
    (101, 1, 3, True),
    (102, 3, 2, True),
    (103, 3, 4, True),
    (104, 6, 7, True),  # disconnected island
    (105, 8, 3, True),
    (106, 8, 9, True),  # orphan (9 off)
    (107, 5, 3, False),  # off -> node 5 isolated
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
# Kernels on bare contract arrays (no model)
# ---------------------------------------------------------------------------


def test_slack_demote_core_on_bare_arrays():
    assert _slack_nodes_to_demote(_bare_nodes(), _bare_branches()) == [2]


def test_gen_promote_core_on_bare_arrays():
    assert _gen_nodes_to_promote(_bare_nodes(), _bare_gens()) == [4]


def test_orphan_branches_core_on_bare_arrays():
    # branch 106 (8->9) points to off node 9.
    assert _orphan_branches_to_disable(_bare_nodes(), _bare_branches()) == [106]


def test_isolated_nodes_core_on_bare_arrays():
    # node 5: the only incident branch 107 is off.
    assert _isolated_nodes_to_disable(_bare_nodes(), _bare_branches()) == [5]


def test_disconnected_nodes_core_on_bare_arrays():
    # islands {5} (isolated) and {6,7} have no active path to slack {1,2}.
    plan = _disconnected_nodes_to_disable(_bare_nodes(), _bare_branches())
    assert set(plan) == {5, 6, 7}
    assert len(plan) == 3  # no duplicates


def test_disconnected_core_no_slack_returns_empty():
    nodes = _bare_nodes()
    nodes["node_type"] = _PQ  # clear all slack
    assert _disconnected_nodes_to_disable(nodes, _bare_branches()) == []


# ---------------------------------------------------------------------------
# Kernel plan == adapter effect on the model (kernel <-> adapter link)
# ---------------------------------------------------------------------------


def _build_model():
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
    """Each adapter changes exactly the ids its kernel returned (on the initial state)."""
    m = _build_model()

    # slack→PQ
    plan = _slack_nodes_to_demote(m.nodes.to_numpy(), m.branches.to_numpy())
    n = refine_slack_to_one(m)
    assert n == {"demoted": len(plan)}
    types = {int(r["id"]): int(r["node_type"]) for r in m.nodes.to_numpy()}
    for nid in plan:
        assert types[nid] == _PQ

    # gen->PV (after refine_slack, as in the pipeline)
    plan = _gen_nodes_to_promote(m.nodes.to_numpy(), m.generators.to_numpy())
    n = refine_node_types_from_generators(m)
    assert n["promoted"] == len(plan)
    types = {int(r["id"]): int(r["node_type"]) for r in m.nodes.to_numpy()}
    for nid in plan:
        assert types[nid] == 1  # PV

    # orphan branches
    plan = _orphan_branches_to_disable(m.nodes.to_numpy(), m.branches.to_numpy())
    n = disable_orphan_branches(m)
    assert n == {"disabled": len(plan)}
    bstat = {int(r["id"]): bool(r["status"]) for r in m.branches.to_numpy()}
    for bid in plan:
        assert bstat[bid] is False

    # disconnected
    plan = _disconnected_nodes_to_disable(m.nodes.to_numpy(), m.branches.to_numpy())
    n = disable_disconnected_components(m)
    assert n == {"disabled": len(plan)}
    nstat = {int(r["id"]): bool(r["status"]) for r in m.nodes.to_numpy()}
    for nid in plan:
        assert nstat[nid] is False

    # isolated
    plan = _isolated_nodes_to_disable(m.nodes.to_numpy(), m.branches.to_numpy())
    n = disable_isolated_nodes(m)
    assert n == {"disabled": len(plan)}


def test_gen_demote_core_pv_without_active_generator():
    """A PV node whose generators are all off returns to PQ; a PV node without a catalogue does not."""
    nodes = np.zeros(3, dtype=SE_INPUT.nodes.input_dtype())
    nodes["id"] = [10, 20, 30]
    nodes["node_type"] = 1  # PV
    nodes["status"] = True
    gens = np.zeros(3, dtype=SE_INPUT.generators.input_dtype())
    gens["id"] = [1, 2, 3]
    gens["node_id"] = [10, 20, 20]
    gens["status"] = [False, False, True]
    # 10: the only generator is off -> PQ; 20: one generator active -> stays PV;
    # 30: no generators at all (PV from the input) -> untouched.
    assert _gen_nodes_to_demote(nodes, gens) == [10]


def test_refine_node_types_order_independent():
    """Node type does not depend on whether the generator is off in the input or switched off by a cascade later."""
    m = _build_model()
    refine_slack_to_one(m)
    refine_node_types_from_generators(m)  # first pass: generator active -> PV
    arr = m.generators.to_numpy().copy()
    arr["status"] = False  # the cascade switched all machines off
    m.generators.update_from_array(arr)
    stats = refine_node_types_from_generators(m)  # second pass
    assert stats["demoted"] >= 1
    assert all(int(r["node_type"]) != 1 for r in m.nodes.to_numpy())
