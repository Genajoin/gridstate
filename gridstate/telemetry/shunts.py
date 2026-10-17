"""Shunts/reactances: reactors -> node.shunt_b, normalization of breaker-branch X.

**Decomposition:** each function is split into an ``_*_on_arrays`` **core over
contract numpy arrays** (mutates the passed arrays in place, returns stats) plus a
thin adapter (``to_numpy().copy()`` -> core -> ``update_from_array``). The early
return for empty reactors/branches stays in the adapter, so ``to_numpy`` is not
called when there is no data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gridstate._arrays import id_value_map
from gridstate.units import BASE_MVA
from gridstate.utils import id_to_pos_map


if TYPE_CHECKING:
    from gridstate.working import Working


# Some input formats replace "breaker" (short-circuit) branches having R=X=0
# with R=0, X=1.0 Ohm. The value is fixed in Ohms, so in p.u. it depends on the
# node's voltage class. The constant must match the input format's sentinel X;
# the name is public: the data producer checks its convention with a cross-test.
BREAKER_X_SENTINEL_OHM = 1.0
#: Default target X of a normalized breaker branch, p.u.
BREAKER_X_PU = 1e-4


def apply_reactors_to_node_shunt(model: Working) -> dict[str, int | float]:
    """Add the B/G of active shunts to ``shunt_b/g`` of their nodes.

    Shunt elements (reactors/capacitor banks) are given by the ``shunts`` table
    (B/G in siemens, with sign and ON_LINE status already applied by the external
    data source). Without this the Y-bus ignores shunt compensation and the
    computed node Q does not account for it.

    Iterates over the active ``shunts`` rows (``status=True``) and, for each one
    whose node is also active, adds its ``B/G`` to the node's ``shunt_b/g``. Apply
    **after** the ON_LINE topology has been applied.

    Returns:
        ``{"applied": N, "sum_b_added_S": float, "sum_g_added_S": float}``.
    """
    shunts_coll = getattr(model, "shunts", None)
    shunts_arr = shunts_coll.to_numpy() if shunts_coll is not None else None
    if shunts_arr is None or len(shunts_arr) == 0:
        return {"applied": 0, "sum_b_added_S": 0.0, "sum_g_added_S": 0.0}
    nodes = model.nodes.to_numpy().copy()
    stats = _aggregate_shunts_on_arrays(nodes, shunts_arr)
    model.nodes.update_from_array(nodes)
    return stats


def _aggregate_shunts_on_arrays(nodes_arr: Any, shunts_arr: Any) -> dict[str, int | float]:
    """Add the B/G of active shunts (``shunts`` table) to ``node.shunt_b/g``.

    ``shunts`` is already in siemens with sign and ON_LINE status applied. Filters
    by ``shunts.status`` AND node activity, sums in row order. Mutates
    ``nodes_arr``. The early return for empty ``shunts`` lives in the adapter
    :func:`apply_reactors_to_node_shunt` (no ``to_numpy`` when there is no data).
    """
    by_id = id_to_pos_map(nodes_arr["id"])
    node_status: dict[int, bool] = id_value_map(nodes_arr, "status", bool)

    applied = 0
    sum_b = 0.0
    sum_g = 0.0
    for s in shunts_arr:
        if not bool(s["status"]):
            continue
        nid = int(s["node_id"])
        if nid == 0 or not node_status.get(nid, False):
            continue
        idx = by_id.get(nid)
        if idx is None:
            continue
        b_add = float(s["susceptance"])
        g_add = float(s["conductance"])
        nodes_arr[idx]["shunt_b"] = float(nodes_arr[idx]["shunt_b"]) + b_add
        nodes_arr[idx]["shunt_g"] = float(nodes_arr[idx]["shunt_g"]) + g_add
        sum_b += b_add
        sum_g += g_add
        applied += 1

    return {"applied": applied, "sum_b_added_S": sum_b, "sum_g_added_S": sum_g}


def normalize_breaker_reactance(
    model: Working, *, eps_pu: float = BREAKER_X_PU
) -> dict[str, int | float]:
    """Bring the X of "breaker" branches to a volt-aware value ``X_pu=eps_pu``.

    The input format replaces branches with ``R=X=0`` (bus sections, breakers,
    block ties) by ``R=0, X=1.0 Ohm`` -- FIXED in Ohms, regardless of the voltage
    class. After ``model_to_pu`` this gives ``X_pu = 1.0 / (Vn²/S_base)``: at
    500 kV ``X_pu≈4e-5`` (fine), but on a 10.5 kV generator-block bus
    ``X_pu≈0.9`` -- the generator P/Q injection drags V down to ~0.79 and a valid
    V measurement is falsely rejected by the solver (``|r/σ|≈6``). Hence the
    β outliers of ``dV_max`` occur precisely on the 6-16 kV LV classes (generator
    block buses after ``aggregate_generators_to_node``).

    The fix (``x=eps_pu`` p.u., NOT ``1.0 Ohm``):
    for each such branch ``X = eps_pu·(Vn²/S_base)`` -> ``X_pu=eps_pu`` on any
    class. The default ``1e-4`` p.u. keeps the voltage drop across a breaker
    negligible on EHV classes too: ``1e-3`` p.u. is 5.6 Ohm at 750 kV and
    drops 2-6 kV at shunt reactor nodes. ``Vn`` is taken from the ``from`` node (or ``to``). ``R=0`` is kept.
    Branches without a defined ``Vn>0`` are left untouched.

    Apply **right after** the model is loaded, before telemetry is applied.
    Reduces false rejection of V measurements on LV classes; on models without
    sentinel branches it is an exact no-op.

    Args:
        model: the working model (Working).
        eps_pu: target ``X_pu`` (default :data:`BREAKER_X_PU`).

    Returns:
        ``{"normalized": N, "eps_pu": eps_pu}``.
    """
    arr = model.branches.to_numpy()
    if arr is None or len(arr) == 0:
        return {"normalized": 0, "eps_pu": eps_pu}

    nodes = model.nodes.to_numpy()
    arr = arr.copy()
    n = _normalize_breaker_reactance_on_arrays(arr, nodes, eps_pu=eps_pu)
    if n:
        model.branches.update_from_array(arr)
    return {"normalized": n, "eps_pu": eps_pu}


def _normalize_breaker_reactance_on_arrays(
    branches_arr: Any, nodes_arr: Any, *, eps_pu: float = BREAKER_X_PU
) -> int:
    """Bring the X of sentinel branches (R=0, X=1.0 Ohm) to ``X_pu=eps_pu`` (mutates ``branches_arr``).

    Reads ``branch.{resistance,reactance,from_node,to_node}`` and
    ``node.{id,voltage_nominal}``; writes ``branch.reactance``. ``Vn`` is taken
    from the ``from`` node (falling back to ``to``); branches without ``Vn>0`` are
    left untouched. Returns the number of normalized branches.
    """
    vn_by_id: dict[int, float] = id_value_map(nodes_arr, "voltage_nominal")

    n = 0
    for i in range(len(branches_arr)):
        r = float(branches_arr[i]["resistance"])
        x = float(branches_arr[i]["reactance"])
        # Loader sentinel: R exactly 0 and X exactly 1.0 Ohm. Real branches
        # always have R>0, so a match uniquely identifies the substituted
        # "breaker" branch.
        # Idempotence: after normalization X = eps_pu·(Vn²/S_base) ≠ 1.0
        # on all classes EXCEPT the coincidence Vn²=S_base/eps_pu (with the
        # default 1e-4 and S_base=100 that is Vn=1000 kV -- not a real class).
        # Even at that point a repeated match recomputes X to the same value
        # (value-idempotent), only inflating the "normalized" counter.
        if abs(r) > 1e-12 or abs(x - BREAKER_X_SENTINEL_OHM) > 1e-9:
            continue
        vn = vn_by_id.get(int(branches_arr[i]["from_node"]), 0.0)
        if vn <= 0.0:
            vn = vn_by_id.get(int(branches_arr[i]["to_node"]), 0.0)
        if vn <= 0.0:
            continue
        branches_arr[i]["reactance"] = eps_pu * (vn * vn / BASE_MVA)
        n += 1

    return n
