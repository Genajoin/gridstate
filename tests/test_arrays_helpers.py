"""Shared structured-array idioms (:mod:`gridstate._arrays`)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate._arrays import (
    active_adjacency,
    active_ids,
    id_value_map,
    meas_mask,
    nodes_with_active_branch,
)
from gridstate.constants import MeasurementObjectType, MeasurementQuality, MeasurementType
from gridstate.contract import SE_INPUT


def _table(dtype: np.dtype, rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _nodes(rows: list[dict]) -> np.ndarray:
    return _table(SE_INPUT.nodes.input_dtype(), rows)


def _branches(rows: list[dict]) -> np.ndarray:
    return _table(SE_INPUT.branches.input_dtype(), rows)


def _meas(rows: list[dict]) -> np.ndarray:
    base = SE_INPUT.measurements.input_dtype()
    fields = [(n, base[n]) for n in base.names]
    if "estimated_si" not in base.names:  # a working-layer column
        fields.append(("estimated_si", np.float64))
    return _table(np.dtype(fields), rows)


_NODE = int(MeasurementObjectType.NODE)
_BRANCH = int(MeasurementObjectType.BRANCH)
_V = int(MeasurementType.VOLTAGE)
_P = int(MeasurementType.POWER_P)
_Q = int(MeasurementType.POWER_Q)


def test_id_value_map_matches_inline_comprehension() -> None:
    nodes = _nodes(
        [
            {"id": 7, "voltage_nominal": 110.0, "status": True},
            {"id": 3, "voltage_nominal": 10.5, "status": False},
            {"id": 7, "voltage_nominal": 220.0, "status": True},  # duplicate: last wins
        ]
    )
    expected = {
        int(i): float(v) for i, v in zip(nodes["id"], nodes["voltage_nominal"], strict=True)
    }
    got = id_value_map(nodes, "voltage_nominal")
    assert got == expected == {7: 220.0, 3: 10.5}
    assert list(got) == list(expected)
    assert all(type(v) is float for v in got.values())
    status = id_value_map(nodes, "status", cast=bool)
    assert status == {7: True, 3: False}
    assert all(type(v) is bool for v in status.values())


def test_active_ids() -> None:
    nodes = _nodes(
        [{"id": 1, "status": True}, {"id": 2, "status": False}, {"id": 5, "status": True}]
    )
    assert active_ids(nodes) == {1, 5}
    assert active_ids(nodes[:0]) == set()


def test_active_adjacency_default_and_mask() -> None:
    br = _branches(
        [
            {"id": 1, "from_node": 1, "to_node": 2, "status": True, "branch_type": 0},
            {"id": 2, "from_node": 2, "to_node": 3, "status": False, "branch_type": 1},
            {"id": 3, "from_node": 3, "to_node": 4, "status": True, "branch_type": 1},
        ]
    )
    adj = active_adjacency(br)
    assert dict(adj) == {1: {2}, 2: {1}, 3: {4}, 4: {3}}
    assert adj[99] == set()  # defaultdict: nodes without branches are empty
    trafo = active_adjacency(br, br["status"].astype(bool) & (br["branch_type"] == 1))
    assert dict(trafo) == {3: {4}, 4: {3}}


def test_active_adjacency_insertion_order_matches_loop() -> None:
    rng = np.random.default_rng(0)
    n = 200
    br = _branches(
        [
            {
                "id": k,
                "from_node": int(rng.integers(0, 40)) * 8,
                "to_node": int(rng.integers(0, 40)) * 8,
                "status": bool(rng.integers(0, 2)),
            }
            for k in range(n)
        ]
    )
    ref: dict[int, set[int]] = {}
    for f, t, st in zip(br["from_node"], br["to_node"], br["status"], strict=True):
        if st:
            ref.setdefault(int(f), set()).add(int(t))
            ref.setdefault(int(t), set()).add(int(f))
    adj = active_adjacency(br)
    assert list(adj) == list(ref)
    for node, nbrs in ref.items():
        assert list(adj[node]) == list(nbrs)


def test_nodes_with_active_branch() -> None:
    br = _branches(
        [
            {"id": 1, "from_node": 1, "to_node": 2, "status": True},
            {"id": 2, "from_node": 3, "to_node": 4, "status": False},
        ]
    )
    assert nodes_with_active_branch(br) == {1, 2}


def _sample_meas() -> np.ndarray:
    return _meas(
        [
            {"id": 1, "object_type": _NODE, "measurement_type": _V, "status": True},
            {
                "id": 2,
                "object_type": _NODE,
                "measurement_type": _V,
                "status": True,
                "is_pseudo": True,
            },
            {"id": 3, "object_type": _BRANCH, "measurement_type": _P, "status": True},
            {"id": 4, "object_type": _BRANCH, "measurement_type": _Q, "status": False},
            {
                "id": 5,
                "object_type": _NODE,
                "measurement_type": _V,
                "status": True,
                "quality": int(MeasurementQuality.BAD),
            },
        ]
    )


def _ids(meas: np.ndarray, mask: np.ndarray) -> list[int]:
    return [int(i) for i in meas["id"][mask]]


def test_meas_mask_filters() -> None:
    m = _sample_meas()
    m["estimated_si"] = [1.0, np.nan, 2.0, 3.0, 4.0]
    assert _ids(m, meas_mask(m)) == [1, 2, 3, 5]
    assert _ids(m, meas_mask(m, active=False)) == [1, 2, 3, 4, 5]
    assert _ids(m, meas_mask(m, pseudo=False)) == [1, 3, 5]
    assert _ids(m, meas_mask(m, pseudo=True)) == [2]
    assert _ids(m, meas_mask(m, kinds=_V, obj=_NODE, pseudo=False)) == [1, 5]
    assert _ids(m, meas_mask(m, kinds=(_P, _Q), active=False)) == [3, 4]
    assert _ids(m, meas_mask(m, exclude_bad=True)) == [1, 2, 3]
    assert _ids(m, meas_mask(m, finite_estimate=True)) == [1, 3, 5]


def test_meas_mask_missing_pseudo_column() -> None:
    m = _sample_meas()
    names = [n for n in m.dtype.names if n != "is_pseudo"]
    bare = np.zeros(len(m), dtype=[(n, m.dtype[n]) for n in names])
    for n in names:
        bare[n] = m[n]
    assert _ids(bare, meas_mask(bare, pseudo=False, missing_pseudo="real")) == [1, 2, 3, 5]
    assert _ids(bare, meas_mask(bare, pseudo=True, missing_pseudo="real")) == []
    assert _ids(bare, meas_mask(bare)) == [1, 2, 3, 5]  # no pseudo filter: column not read
    with pytest.raises((KeyError, ValueError)):
        meas_mask(bare, pseudo=False)
