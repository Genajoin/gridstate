"""Topological operations on the network model — ported into gridstate.

The topological clean-up is implemented entirely inside gridstate, without external
dependencies (the equivalence guard is ``tests/test_topology_port.py``).

**Decision kernel over contract arrays.** Each function is split into:

* ``_*`` **pure kernel** — takes contract numpy arrays
  (``model.nodes.to_numpy()`` etc., or "bare" arrays ``SE_INPUT.*.
  input_dtype()``), reads ONLY contract columns and returns a **plan** —
  a list of object ``id``s whose status/type must be changed. The kernel does not depend on
  ``Working`` or on the way of writing;
* public **adapter** function — materializes the arrays from the model, calls the kernel,
  applies the plan with the existing object-level ``model.{coll}.update(id, {...})``
  (the write path is unchanged), and returns a counter.

The "pure kernel ↔ adapter" split makes it possible to change the way the plan is applied
(object-level ``model.update`` or assignment into the working array) without touching the logic
of the kernel itself.

Families:

1. ``disable_*`` — switching off nodes/branches by topological criteria
   (orphan, disconnected, isolated). Universal for any format.
2. ``refine_*`` — bringing node types to the semantics of the solvers
   (multiple SLACK→one; promote PQ→PV by the presence of generators).

Contract columns (see ``gridstate.contract.tables``, role WORKING): the kernels read
``node.{id,status,node_type,balance_priority}``, ``branch.{id,status,from_node,
to_node}``, ``generator.{status,node_id}``; the adapters write ``node.{status,
node_type}``, ``branch.status``.

Typical order of preparing a model for the numerical solution::

    refine_slack_to_one(model)
    refine_node_types_from_generators(model)
    disable_orphan_branches(model)
    disable_disconnected_components(model)
    disable_isolated_nodes(model)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from gridstate._arrays import active_ids, nodes_with_active_branch
from gridstate.constants import NodeType


# Plain ints for hot comparison loops; values come from the canonical enum.
_SLACK_NODE_TYPE = int(NodeType.SLACK)
_PQ_NODE_TYPE = int(NodeType.PQ)
_PV_NODE_TYPE = int(NodeType.PV)


# ---------------------------------------------------------------------------
# Pure kernels over contract arrays (return an id plan)
# ---------------------------------------------------------------------------


def _orphan_branches_to_disable(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Branches referring to a missing/disabled node → a disable plan.

    Reads ``node.{id,status}`` and ``branch.{id,status,from_node,to_node}``.
    Returns the ``id``s of active branches with at least one end not in the set of
    active nodes (order — as in ``branches_arr``).
    """
    valid_node_ids = active_ids(nodes_arr)

    to_disable: list[int] = []
    for r in branches_arr:
        if not r["status"]:
            continue
        f = int(r["from_node"])
        t = int(r["to_node"])
        if f not in valid_node_ids or t not in valid_node_ids:
            to_disable.append(int(r["id"]))
    return to_disable


def _disconnected_nodes_to_disable(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Nodes without an active path to slack → a disable plan (BFS from the slack nodes).

    Reads ``node.{id,status,node_type}`` and ``branch.{status,from_node,to_node}``.
    If there are no slack nodes — returns ``[]`` (as the original: 0 without modifications).
    """
    active_node_ids: set[int] = set()
    adj: dict[int, list[int]] = {}
    slack_seeds: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        nid = int(r["id"])
        active_node_ids.add(nid)
        adj[nid] = []
        if int(r["node_type"]) == _SLACK_NODE_TYPE:
            slack_seeds.append(nid)

    for r in branches_arr:
        if not r["status"]:
            continue
        f = int(r["from_node"])
        t = int(r["to_node"])
        if f in adj and t in adj:
            adj[f].append(t)
            adj[t].append(f)

    if not slack_seeds:
        return []

    visited: set[int] = set(slack_seeds)
    queue: deque[int] = deque(slack_seeds)
    while queue:
        nid = queue.popleft()
        for nb in adj.get(nid, ()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return list(active_node_ids - visited)


def _isolated_nodes_to_disable(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Active nodes without a single active branch → a disable plan (slack is not touched).

    Reads ``node.{id,status,node_type}`` and ``branch.{status,from_node,to_node}``.
    """
    has_active_branch = nodes_with_active_branch(branches_arr)

    to_disable: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        if int(r["node_type"]) == _SLACK_NODE_TYPE:
            continue
        nid = int(r["id"])
        if nid not in has_active_branch:
            to_disable.append(nid)
    return to_disable


def _slack_nodes_to_demote(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Reduce the set of SLACK nodes to one → a plan to demote the rest to PQ.

    Reads ``node.{id,status,node_type,balance_priority}`` and
    ``branch.{status,from_node,to_node}``. Candidates — active SLACK with
    ``balance_priority>0`` and an active branch; keep the min priority (ties → min
    id), the rest SLACK → PQ. With no valid candidates — ``[]``.

    The slack picked by ``units.network_pu_from_tables`` when several remain
    follows a different rule (any balance_priority, no branch check, ties to
    the first position); the two coincide only when this plan leaves one slack.
    """
    nodes_with_branch = nodes_with_active_branch(branches_arr)

    candidates: list[tuple[int, int]] = []  # (balance_priority, node_id)
    for r in nodes_arr:
        if not r["status"]:
            continue
        if int(r["node_type"]) != _SLACK_NODE_TYPE:
            continue
        nid = int(r["id"])
        bp = int(r["balance_priority"])
        if bp <= 0:
            continue
        if nid not in nodes_with_branch:
            continue
        candidates.append((bp, nid))

    if not candidates:
        return []  # nothing is demoted — no valid candidates

    candidates.sort()  # min priority first; ties → min id
    chosen_slack = candidates[0][1]

    to_demote: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        if int(r["node_type"]) != _SLACK_NODE_TYPE:
            continue
        nid = int(r["id"])
        if nid != chosen_slack:
            to_demote.append(nid)
    return to_demote


def _gen_nodes_to_promote(nodes_arr: Any, generators_arr: Any) -> list[int]:
    """Nodes with an active generator → a PQ→PV plan.

    Reads ``generator.{status,node_id}`` and ``node.{id,status,node_type}``:
    any PQ node with an active generator.
    """
    has_active_gen: set[int] = set()
    for r in generators_arr:
        if r["status"]:
            has_active_gen.add(int(r["node_id"]))

    to_promote: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        nid = int(r["id"])
        if nid not in has_active_gen:
            continue
        # SLACK is not demoted to PV; only PQ → PV.
        if int(r["node_type"]) == _PQ_NODE_TYPE:
            to_promote.append(nid)
    return to_promote


def _gen_nodes_to_demote(nodes_arr: Any, generators_arr: Any) -> list[int]:
    """PV nodes that have generators but none of them active → a PV→PQ plan.

    A pair to :func:`_gen_nodes_to_promote`: the first pass over types runs before the
    ``node off ⇒ gen off`` cascade and promotes a node by a generator that then
    goes dark. We demote only nodes with generator records — a PV that came from an
    input without a generator catalogue is not touched.
    """
    has_gen: set[int] = set()
    has_active_gen: set[int] = set()
    for r in generators_arr:
        nid = int(r["node_id"])
        has_gen.add(nid)
        if r["status"]:
            has_active_gen.add(nid)
    return [
        int(r["id"])
        for r in nodes_arr
        if int(r["node_type"]) == _PV_NODE_TYPE
        and int(r["id"]) in has_gen
        and int(r["id"]) not in has_active_gen
    ]


# ---------------------------------------------------------------------------
# Public adapters: arrays from the model → kernel → apply the plan via .update()
# ---------------------------------------------------------------------------


def disable_orphan_branches(model: Any) -> dict[str, int]:
    """Disable branches referring to missing or disabled nodes.

    Real test models contain branches whose ``from_node`` or ``to_node``
    is not in the node table (the model was exported for a sub-area without cleaning out
    the boundary branches). Such a branch is formally ``status=True``, but one of its
    ends is unavailable — it is a "dangling" branch that breaks the topology.

    Algorithm: for each active branch it is checked whether both of its nodes are in the set of
    active nodes. If at least one is missing or disabled, the branch is disabled.

    Returns:
        Step statistics ``{"disabled": <number of disabled branches>}``.

    Note:
        Apply **before** ``disable_isolated_nodes``, otherwise the neighbour nodes of
        orphan branches would be considered connected.
    """
    to_disable = _orphan_branches_to_disable(model.nodes.to_numpy(), model.branches.to_numpy())
    for bid in to_disable:
        model.branches.update(bid, {"status": False})
    return {"disabled": len(to_disable)}


def disable_disconnected_components(model: Any) -> dict[str, int]:
    """Disable nodes not connected to slack through active branches.

    From the set of active nodes and active branches a BFS is run from all slack nodes
    (``node_type == 2``). Nodes not reached by the BFS are disabled.

    Why it is needed: ``disable_isolated_nodes`` finds nodes **without** active branches.
    But **isolated subnetworks** remain — nodes connected to each other by an active
    branch but without an active path to slack. Their δ angles are free degrees of
    freedom of H, which gives a rank deficiency of the SE / Power Flow Jacobian.

    Returns:
        Step statistics ``{"disabled": <number of disabled nodes>}``.

    Note:
        - Apply **after** ``disable_orphan_branches`` and **before**
          ``disable_isolated_nodes``.
        - If there are no slack nodes, the function returns 0 without modifications.
        - There may be several slack nodes (one per island) — all are starting points.
    """
    to_disable = _disconnected_nodes_to_disable(model.nodes.to_numpy(), model.branches.to_numpy())
    for nid in to_disable:
        model.nodes.update(nid, {"status": False})
    return {"disabled": len(to_disable)}


def disable_isolated_nodes(model: Any) -> dict[str, int]:
    """Disable nodes left without active branches.

    In real models some nodes are marked ``status=True`` but **all**
    incident branches have ``status=False``. Such a node is an isolated island: its
    angle δ is unobservable, which makes the H matrix singular.

    Slack nodes (``node_type == 2``) are **not disabled**, even if isolated —
    this is a signal of a topology error, not a silent "fix".

    Returns:
        Step statistics ``{"disabled": <number of disabled nodes>}``.

    Note:
        It is recommended to call ``disable_orphan_branches`` first.
    """
    to_disable = _isolated_nodes_to_disable(model.nodes.to_numpy(), model.branches.to_numpy())
    for nid in to_disable:
        model.nodes.update(nid, {"status": False})
    return {"disabled": len(to_disable)}


def refine_slack_to_one(model: Any) -> dict[str, int]:
    """Reduce the set of SLACK nodes to one via ``balance_priority``.

    The input may flag several nodes as balancing (typically 5-15 in a production
    model). A solver that picks its own slack tolerates that, but for a typical
    PF/SE several slacks mean several islands with independent δ, and convergence
    breaks.

    Algorithm:
    1. Candidates — active SLACK (``node_type=2``) with ``balance_priority>0`` and
       active-branch.
    2. Keep the slack with the minimal ``balance_priority`` (ties → min id).
    3. The remaining slack nodes are demoted to PQ.

    Apply **right after loading**, BEFORE ``refine_node_types_from_generators``.

    Returns:
        Step statistics ``{"demoted": <number of SLACK→PQ nodes>}``.
    """
    to_demote = _slack_nodes_to_demote(model.nodes.to_numpy(), model.branches.to_numpy())
    for nid in to_demote:
        model.nodes.update(nid, {"node_type": _PQ_NODE_TYPE})
    return {"demoted": len(to_demote)}


def refine_node_types_from_generators(model: Any) -> dict[str, int]:
    """Mark nodes with active generators as PV (``node_type=1``).

    A node all of whose generators are switched off returns from PV to PQ —
    otherwise the type would depend on whether the machines are off in the input or were switched off by the
    pipeline cascade after the first pass.

    The input format does **not classify PV nodes** — all except SLACK are marked PQ;
    any PQ node with an active generator becomes a PV node.

    Apply **after** ``refine_slack_to_one``.

    Returns:
        Step statistics ``{"promoted": <PQ→PV>, "demoted": <PV→PQ>}``.
    """
    nodes_arr = model.nodes.to_numpy()
    gens_arr = model.generators.to_numpy()
    to_demote = _gen_nodes_to_demote(nodes_arr, gens_arr)
    to_promote = _gen_nodes_to_promote(nodes_arr, gens_arr)
    for nid in to_demote:
        model.nodes.update(nid, {"node_type": _PQ_NODE_TYPE})
    for nid in to_promote:
        model.nodes.update(nid, {"node_type": _PV_NODE_TYPE})
    return {"promoted": len(to_promote), "demoted": len(to_demote)}


__all__ = [
    "disable_disconnected_components",
    "disable_isolated_nodes",
    "disable_orphan_branches",
    "refine_node_types_from_generators",
    "refine_slack_to_one",
]
