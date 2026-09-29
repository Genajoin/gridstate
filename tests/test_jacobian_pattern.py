"""The fixed-pattern Jacobian is bit-identical to the generic sparse path."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.algebra.base import BaseAlgebra
from gridstate.state import StateLayout
from gridstate.units import model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_QNAG,
    KIND_CURRENT,
    KIND_NODE_BALANCE_P,
    KIND_NODE_BALANCE_Q,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    SIDE_FROM,
    SIDE_NONE,
    SIDE_TO,
    MeasurementIndex,
)
from tests.test_algebra_base import _build_three_bus


def _index(rows: list[tuple[int, int, int, int]]) -> MeasurementIndex:
    arr = np.array(rows, dtype=np.int64)
    return MeasurementIndex(
        kind=arr[:, 0],
        object_kind=arr[:, 1],
        object_pos=arr[:, 2],
        branch_side=arr[:, 3],
        meas_id=np.arange(len(rows), dtype=np.int64) + 1,
    )


def _algebra(with_box: bool) -> BaseAlgebra:
    pu = model_to_pu(_build_three_bus())
    ybus, yf, yt = build_ybus(pu)
    rows = []
    for b in range(pu.n_bus):
        rows += [
            (KIND_VOLTAGE, OBJ_NODE, b, SIDE_NONE),
            (KIND_POWER_INJECTION_P, OBJ_NODE, b, SIDE_NONE),
            (KIND_POWER_INJECTION_Q, OBJ_NODE, b, SIDE_NONE),
        ]
    for br in range(pu.n_branch):
        for side in (SIDE_FROM, SIDE_TO):
            rows += [
                (KIND_POWER_P, OBJ_BRANCH, br, side),
                (KIND_POWER_Q, OBJ_BRANCH, br, side),
                (KIND_CURRENT, OBJ_BRANCH, br, side),
            ]
    non_slack = np.array([i for i in range(pu.n_bus) if i != pu.slack_idx], dtype=np.int64)
    if with_box:
        box_bus = int(non_slack[0])
        rows += [
            (KIND_NODE_BALANCE_P, OBJ_NODE, box_bus, SIDE_NONE),
            (KIND_NODE_BALANCE_Q, OBJ_NODE, box_bus, SIDE_NONE),
            (KIND_BOX_PRIOR_PGEN, OBJ_NODE, box_bus, SIDE_NONE),
            (KIND_BOX_PRIOR_QNAG, OBJ_NODE, box_bus, SIDE_NONE),
        ]
        pos = np.array([box_bus], dtype=np.int64)
        layout = StateLayout(
            n_bus=pu.n_bus,
            slack_idx=pu.slack_idx,
            non_slack_idx=non_slack,
            pgen_node_pos=pos,
            qgen_node_pos=pos,
            pnag_node_pos=pos,
            qnag_node_pos=pos,
        )
    else:
        layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    # Shuffle the rows: the pattern must follow the order of z, not of kinds.
    order = np.random.default_rng(7).permutation(len(rows))
    return BaseAlgebra(ybus, yf, yt, _index([rows[i] for i in order]), layout, pu)


def _canonical(H):
    H = H.tocsr(copy=True)
    H.eliminate_zeros()
    H.sort_indices()
    return H


@pytest.mark.parametrize("with_box", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pattern_matches_generic_path_bit_for_bit(with_box: bool, seed: int) -> None:
    algebra = _algebra(with_box)
    rng = np.random.default_rng(seed)
    v = 1.0 + rng.uniform(-0.08, 0.08, algebra.n_bus)
    delta = rng.uniform(-0.3, 0.3, algebra.n_bus)
    delta[algebra.layout.slack_idx] = 0.0

    for _ in range(2):  # the second call reuses the cached pattern
        fast = _canonical(algebra.evaluate_jacobian(v, delta))
        generic = _canonical(algebra._evaluate_jacobian_generic(v, delta))
        assert fast.shape == generic.shape
        assert np.array_equal(fast.indptr, generic.indptr)
        assert np.array_equal(fast.indices, generic.indices)
        assert np.array_equal(fast.data, generic.data)
    assert algebra._jacobian_pattern not in (None, False)


def test_unsupported_combination_falls_back_to_generic_path() -> None:
    pu = model_to_pu(_build_three_bus())
    ybus, yf, yt = build_ybus(pu)
    # A branch flow without a side is not covered by the pattern.
    idx = _index([(KIND_VOLTAGE, OBJ_NODE, 0, SIDE_NONE), (KIND_POWER_P, OBJ_BRANCH, 0, SIDE_NONE)])
    algebra = BaseAlgebra(ybus, yf, yt, idx, StateLayout.from_slack(pu.n_bus, pu.slack_idx), pu)
    v, delta = np.ones(pu.n_bus), np.zeros(pu.n_bus)
    with pytest.raises(ValueError, match="пропущены позиции"):
        algebra.evaluate_jacobian(v, delta)
    assert algebra._jacobian_pattern is False
