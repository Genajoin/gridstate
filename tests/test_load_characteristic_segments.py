"""Piecewise static load characteristics (``load_characteristic_segments``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT, SEInput
from gridstate.contract.serialize import load_se_input_npz, save_se_input
from gridstate.contract.validate import validate_input
from gridstate.load_characteristics import (
    QUANTITY_P,
    QUANTITY_Q,
    segment_factor,
    segment_plan,
    segment_problems,
)
from gridstate.post_processing import apply_load_characteristic
from gridstate.working import Working, _ArrayCollection


# Q: three segments with boundaries 0.815 and 1.2 (constant factors 0.5 / 1 / 2
# plus a V⁴ term on the upper one to exercise the high powers).
_Q_SEGMENTS = [
    (-np.inf, (0.5, 0, 0, 0, 0)),
    (0.815, (1.0, 0, 0, 0, 0)),
    (1.2, (1.0, 0, 0, 0, 1.0)),
]


def _segments(rows) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.load_characteristic_segments.input_dtype())
    for i, (lm, q, v_lo, c) in enumerate(rows):
        arr[i]["load_model_id"] = lm
        arr[i]["quantity"] = q
        arr[i]["v_lo"] = v_lo
        for k in range(5):
            arr[i][f"c{k}"] = c[k]
    return arr


def _q_rows(lm: int = 0):
    return [(lm, QUANTITY_Q, v, c) for v, c in _Q_SEGMENTS]


@pytest.mark.parametrize(
    ("v", "factor"),
    [
        (0.1, 0.5),
        (0.815, 0.5),  # on a boundary: the lower segment
        (0.8151, 1.0),
        (1.2, 1.0),
        (1.3, 1.0 + 1.3**4),
    ],
)
def test_segment_selection_takes_the_lower_segment_on_a_boundary(v, factor):
    plan = segment_plan(_segments(_q_rows()), n_models=1)
    assert segment_factor(*plan[(0, QUANTITY_Q)], v) == pytest.approx(factor)


def test_first_segment_extends_to_zero_whatever_its_v_lo():
    rows = _q_rows()
    rows[0] = (0, QUANTITY_Q, 0.5, rows[0][3])
    plan = segment_plan(_segments(rows), n_models=1)
    assert segment_factor(*plan[(0, QUANTITY_Q)], 0.2) == pytest.approx(0.5)


def test_rows_may_come_unsorted():
    plan = segment_plan(_segments(list(reversed(_q_rows()))), n_models=1)
    assert segment_factor(*plan[(0, QUANTITY_Q)], 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_q_rows(lm=3), "not in load_characteristics"),
        ([(0, 2, -np.inf, (1, 0, 0, 0, 0))], "quantity"),
        ([*_q_rows(), (0, QUANTITY_Q, 1.2, (1, 0, 0, 0, 0))], "strictly increasing"),
        (
            [(0, QUANTITY_P, -np.inf, (1, 0, 0, 0, 0)), (0, QUANTITY_P, np.inf, (1, 0, 0, 0, 0))],
            "finite",
        ),
        ([(0, QUANTITY_P, -np.inf, (np.nan, 0, 0, 0, 0))], "non-finite"),
    ],
)
def test_contract_violations(rows, message):
    seg = _segments(rows)
    assert any(message in p for p in segment_problems(seg, n_models=1))
    with pytest.raises(ValueError, match="load_characteristic_segments"):
        segment_plan(seg, n_models=1)


def _model(segments: np.ndarray | None) -> Working:
    m = Working.empty()
    lc = np.zeros(1, dtype=SE_INPUT.load_characteristics.input_dtype())
    lc["coeff_p_a0"] = 1.0  # quadratic rows: PQ-const
    lc["coeff_q_b0"] = 1.0
    m.load_characteristics = _ArrayCollection(lc)
    if segments is not None:
        m.load_characteristic_segments = _ArrayCollection(segments)
    for nid, v, p0, q0 in [(1, 0.80, 100.0, 50.0), (2, 1.30, 100.0, 50.0), (3, 1.30, 100.0, -5.0)]:
        m.nodes.add(
            {
                "id": nid,
                "status": True,
                "exist_load": 1,
                "load_model_id": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0 * v,
                "load_p": p0,
                "load_q": q0,
                "load_p_estimated": 7.0,
                "load_q_estimated": 3.0,
            }
        )
    return m


def _estimates(m: Working) -> dict[int, tuple[float, float]]:
    arr = m.nodes.to_numpy()
    return {int(r["id"]): (float(r["load_p_estimated"]), float(r["load_q_estimated"])) for r in arr}


def test_piecewise_q_replaces_the_quadratic_row_and_keeps_p():
    m = _model(_segments(_q_rows()))
    out = apply_load_characteristic(m)
    est = _estimates(m)
    assert est[1] == pytest.approx((100.0, 25.0))  # Q on the lower segment
    assert est[2] == pytest.approx((100.0, 50.0 * (1.0 + 1.3**4)))
    # Q0 ≤ 0: the piecewise Q is not applied, the estimate is kept.
    assert est[3] == pytest.approx((100.0, 3.0))
    assert (out["piecewise_p"], out["piecewise_q"]) == (0, 2)


def test_without_segments_the_quadratic_path_is_unchanged():
    m = _model(None)
    apply_load_characteristic(m)
    est = _estimates(m)
    # PQ-const rows, applied whatever the sign of Q0 (previous behaviour).
    assert est == {1: (100.0, 50.0), 2: (100.0, 50.0), 3: (100.0, -5.0)}


def test_roundtrip_and_validation(tmp_path):
    seg = _segments(_q_rows())
    m = _model(seg)
    p = save_se_input(SEInput(model=m), tmp_path / "seg.npz")
    got = load_se_input_npz(p).model.load_characteristic_segments.to_numpy()
    assert got.tobytes() == seg.tobytes()
    assert validate_input(m).ok

    bad = _model(_segments(_q_rows(lm=5)))
    issues = validate_input(bad).issues
    assert [i.kind for i in issues] == ["invalid_rows"]


def test_mapping_source_may_omit_the_table():
    tables = {t.name: np.zeros(0, dtype=t.input_dtype()) for t in SE_INPUT.tables()}
    del tables["load_characteristic_segments"]
    assert validate_input(tables).ok
