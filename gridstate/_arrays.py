"""Read-only idioms over contract structured arrays (nodes, branches, measurements).

Collects the one-liners that were copy-pasted across the preparation layers:
id -> column maps, the set of active ids, adjacency over active branches and the
measurement selection mask. Every helper reproduces the former inline code
exactly (same values, same iteration and insertion order), so replacing a copy
is bit-identical. The id -> position map and the branch endpoint map live in
:mod:`gridstate.utils`.

The measurement mask keeps the call-site differences explicit instead of
unifying them: whether inactive rows count (``active``), real/pseudo selection
(``pseudo``), the behaviour when the ``is_pseudo`` column is absent
(``missing_pseudo``), BAD quality (``exclude_bad``) and a finite estimate
(``finite_estimate``) are all keyword options.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from gridstate.constants import MeasurementQuality


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


_T = TypeVar("_T")

__all__ = [
    "active_adjacency",
    "active_ids",
    "id_value_map",
    "meas_mask",
    "nodes_with_active_branch",
]


@overload
def id_value_map(arr: np.ndarray, col: str) -> dict[int, float]: ...


@overload
def id_value_map(arr: np.ndarray, col: str, cast: Callable[[Any], _T]) -> dict[int, _T]: ...


def id_value_map(arr: np.ndarray, col: str, cast: Callable[[Any], Any] = float) -> dict[int, Any]:
    """Map ``int(arr["id"])`` -> ``cast(arr[col])`` in row order.

    Duplicate ids resolve to the last row, like the dict comprehensions this
    replaces. ``cast`` is ``float`` by default; pass ``bool`` for status columns.
    """
    return {int(i): cast(v) for i, v in zip(arr["id"].tolist(), arr[col].tolist(), strict=True)}


def active_ids(arr: np.ndarray) -> set[int]:
    """Ids of the rows whose ``status`` is truthy (row order of insertion)."""
    return {int(i) for i, s in zip(arr["id"].tolist(), arr["status"].tolist(), strict=True) if s}


def active_adjacency(
    branches: np.ndarray, mask: np.ndarray | None = None
) -> defaultdict[int, set[int]]:
    """Undirected node adjacency over the selected branches.

    ``mask`` selects the branches (default: truthy ``status``). Edges are
    inserted in branch row order, ``from -> to`` then ``to -> from``, so the
    neighbour sets match the former inline loops element for element. A
    ``defaultdict(set)`` is returned: callers index nodes without branches.
    """
    sel = branches["status"].astype(bool) if mask is None else np.asarray(mask, dtype=bool)
    adj: defaultdict[int, set[int]] = defaultdict(set)
    for f, t in zip(
        branches["from_node"][sel].tolist(), branches["to_node"][sel].tolist(), strict=True
    ):
        fi, ti = int(f), int(t)
        adj[fi].add(ti)
        adj[ti].add(fi)
    return adj


def nodes_with_active_branch(branches: np.ndarray) -> set[int]:
    """Nodes at either end of at least one active (``status``) branch."""
    out: set[int] = set()
    for f, t, s in zip(
        branches["from_node"].tolist(),
        branches["to_node"].tolist(),
        branches["status"].tolist(),
        strict=True,
    ):
        if not s:
            continue
        out.add(int(f))
        out.add(int(t))
    return out


def meas_mask(
    meas: np.ndarray,
    *,
    kinds: int | Iterable[int] | None = None,
    obj: int | None = None,
    active: bool = True,
    pseudo: bool | None = None,
    missing_pseudo: Literal["raise", "real"] = "raise",
    exclude_bad: bool = False,
    finite_estimate: bool = False,
) -> np.ndarray:
    """Boolean row mask over a measurements array.

    Args:
        meas: measurements structured array.
        kinds: ``measurement_type`` filter, one code or several (``None``: any).
        obj: ``object_type`` filter (``None``: any).
        active: keep only rows with a truthy ``status``.
        pseudo: ``False`` keeps real rows, ``True`` keeps pseudo rows, ``None``
            does not look at ``is_pseudo``.
        missing_pseudo: what ``pseudo`` does when the array has no
            ``is_pseudo`` column: ``"raise"`` indexes it anyway (the structured
            array raises), ``"real"`` treats every row as real.
        exclude_bad: drop rows with ``quality == BAD``.
        finite_estimate: keep only rows with a finite ``estimated_si``.
    """
    mask: np.ndarray = meas["status"].astype(bool) if active else np.ones(len(meas), dtype=bool)
    if pseudo is not None:
        names = meas.dtype.names or ()
        if missing_pseudo == "raise" or "is_pseudo" in names:
            is_pseudo = meas["is_pseudo"].astype(bool)
            mask &= is_pseudo if pseudo else ~is_pseudo
        elif pseudo:
            mask &= False
    if obj is not None:
        mask &= meas["object_type"] == obj
    if kinds is not None:
        if isinstance(kinds, (int, np.integer)):
            mask &= meas["measurement_type"] == kinds
        else:
            mask &= np.isin(meas["measurement_type"], list(kinds))
    if exclude_bad:
        mask &= meas["quality"].astype(np.int64) != int(MeasurementQuality.BAD)
    if finite_estimate:
        mask &= np.isfinite(meas["estimated_si"])
    return mask
