"""Measurement id allocation for rows added by gridstate (:mod:`gridstate.measurement_ids`)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT
from gridstate.measurement_ids import (
    ANTI_OVERSHOOT_ID_BASE,
    MIRROR_V_ID_BASE,
    PSEUDO_ID_BASE,
    MAX_MEASUREMENT_ID,
    IdAllocator,
    MeasurementIdCollisionError,
)
from gridstate.preprocessing.mirror_voltage import _mirror_voltage_on_arrays
from gridstate.preprocessing.pseudo_measurements import (
    PseudoMeasConfig,
    _add_pseudo_measurements_on_arrays,
)
from gridstate.telemetry.apply_resolved import (
    TelemetryApplyConfig,
    _build_node_injection_rows,
)
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


def _table(dtype: np.dtype, rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def test_bases_are_unchanged() -> None:
    assert MIRROR_V_ID_BASE == 200_000_000
    assert PSEUDO_ID_BASE == 300_000_000
    assert ANTI_OVERSHOOT_ID_BASE == 760_000_000


def test_sequential_ids_from_start() -> None:
    ids = IdAllocator([1, 2, 3], 10, name="t")
    assert [ids.next() for _ in range(3)] == [10, 11, 12]


def test_taken_ids_are_skipped() -> None:
    ids = IdAllocator(np.array([10, 11, 13, 15]), 10, name="t")
    assert [ids.next() for _ in range(3)] == [12, 14, 16]


def test_exhausted_id_space_raises() -> None:
    ids = IdAllocator([MAX_MEASUREMENT_ID], MAX_MEASUREMENT_ID - 1, name="step")
    assert ids.next() == MAX_MEASUREMENT_ID - 1
    with pytest.raises(MeasurementIdCollisionError, match="step: no free measurement id"):
        ids.next()
    assert issubclass(MeasurementIdCollisionError, ValueError)


def test_after_max() -> None:
    assert IdAllocator.after_max(np.array([5, 42, 7]), name="t").next() == 43
    assert IdAllocator.after_max(np.array([], dtype=np.int64), name="t").next() == 1


def test_telemetry_injection_ids_follow_max() -> None:
    meas = _table(SE_INPUT.measurements.input_dtype(), [{"id": 7}, {"id": 3}])
    inj = {(1, "P"): [(10.0, "g1", 0)], (1, "Q"): [(2.0, "g1", 0)]}
    stats = {"node_inj_added": 0, "applied_questionable": 0}
    rows = _build_node_injection_rows(inj, meas, TelemetryApplyConfig(), stats)
    assert [r["id"] for r in rows] == [8, 9]


def _nodes(n: int) -> np.ndarray:
    return _table(
        SE_INPUT.nodes.input_dtype(),
        [{"id": i + 1, "status": True, "voltage_nominal": 110.0} for i in range(n)],
    )


def test_pseudo_ids_step_over_input_ids() -> None:
    # Hashed input ids (CIM) may land inside the pseudo range: they are stepped over.
    nodes = _nodes(2)
    branches = _table(SE_INPUT.branches.input_dtype(), [])
    meas = _table(SE_INPUT.measurements.input_dtype(), [{"id": PSEUDO_ID_BASE + 1}])
    cfg = PseudoMeasConfig(add_zero_injections=False)
    rows, _ = _add_pseudo_measurements_on_arrays(nodes, branches, meas, None, cfg)
    assert [r["id"] for r in rows] == [PSEUDO_ID_BASE, PSEUDO_ID_BASE + 2]
    ok = _table(SE_INPUT.measurements.input_dtype(), [{"id": 5}])
    rows, _ = _add_pseudo_measurements_on_arrays(nodes, branches, ok, None, cfg)
    assert [r["id"] for r in rows] == [PSEUDO_ID_BASE, PSEUDO_ID_BASE + 1]


def test_mirror_ids_step_over_taken() -> None:
    nodes = _nodes(4)
    branches = _table(
        SE_INPUT.branches.input_dtype(),
        [
            {
                "id": 1,
                "from_node": 1,
                "to_node": 2,
                "status": True,
                "branch_type": 1,
                "tap_ratio": 1.0,
            },
            {
                "id": 2,
                "from_node": 3,
                "to_node": 4,
                "status": True,
                "branch_type": 1,
                "tap_ratio": 1.0,
            },
        ],
    )
    real_v = {
        "object_type": OBJ_NODE,
        "measurement_type": KIND_VOLTAGE,
        "status": True,
        "value": 115.0,
        "variance": 0.5,
    }
    meas = _table(
        SE_INPUT.measurements.input_dtype(),
        [
            {**real_v, "id": MIRROR_V_ID_BASE, "object_id": 1},
            {**real_v, "id": MIRROR_V_ID_BASE + 2, "object_id": 3},
        ],
    )
    # The start and the id after the first mirror are taken: both are stepped over.
    rows = _mirror_voltage_on_arrays(nodes, branches, meas)
    assert [r["id"] for r in rows] == [MIRROR_V_ID_BASE + 1, MIRROR_V_ID_BASE + 3]
