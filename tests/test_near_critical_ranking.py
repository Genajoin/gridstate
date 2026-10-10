"""Nearly critical measurements are left out of the residual rankings.

Their ``r_N`` is numerically unreliable (``Ω_ii`` is a small difference of
nearly equal numbers), so ``worst_residuals`` and the normalized-residual
report rank the other measurements and list these apart. A near-critical
threshold of ``0`` restores the plain ranking by ``r_N``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import gridstate.algebra.normalized_residuals as nr
from gridstate.api import estimate
from gridstate.contract.serialize import load_se_input_npz
from gridstate.quality_summary import near_critical_measurements, top_worst_residuals
from gridstate.units import model_to_pu
from gridstate.validation.bad_data import compute_normalized_residuals_report
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    MeasurementIndex,
)


_CASE14 = Path(__file__).parent / "test_data" / "ieee" / "case14.npz"


def _case14_with_near_critical_flow():
    """IEEE case14 where a radial node's angle rests on one precise P flow.

    The P injections at the radial node and its neighbour and the Q flows of
    the radial branch are switched off. Of its two P flows one keeps its
    variance and the other gets a variance 10⁶ times larger and a value off
    by five of its σ. The precise flow is then nearly critical
    (``Ω_ii / σ²_i ≈ 1e-6``), and both flows have ``r_N ≈ 5``, the largest
    of the case. Returns ``(model, near_critical_id, loose_id, branch_id)``.
    """
    model = load_se_input_npz(_CASE14).model
    npu = model_to_pu(model)
    degree = np.bincount(np.concatenate([npu.from_idx, npu.to_idx]), minlength=npu.n_bus)
    radial = int(np.where(degree == 1)[0][0])
    br = int(np.where((npu.from_idx == radial) | (npu.to_idx == radial))[0][0])
    nb = int(npu.to_idx[br] if npu.from_idx[br] == radial else npu.from_idx[br])
    node_ids = [int(npu.bus_ids[radial]), int(npu.bus_ids[nb])]
    branch_id = int(npu.branch_ids[br])

    arr = model.measurements.to_numpy()
    kind = arr["measurement_type"]
    on_node = (arr["object_type"] == OBJ_NODE) & np.isin(arr["object_id"], node_ids)
    on_branch = (arr["object_type"] == OBJ_BRANCH) & (arr["object_id"] == branch_id)
    arr["status"][on_node & np.isin(kind, (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q))] = False
    arr["status"][on_branch & (kind == KIND_POWER_Q)] = False
    precise, loose = np.where(on_branch & (kind == KIND_POWER_P))[0][:2]
    arr["variance"][loose] *= 1e6
    arr["value"][loose] += 5.0 * np.sqrt(arr["variance"][loose])
    model.measurements.update_from_array(arr)
    return model, int(arr["id"][precise]), int(arr["id"][loose]), branch_id


def _solve(**kwargs):
    model, precise, loose, branch_id = _case14_with_near_critical_flow()
    res = estimate(model, algorithm="wls", **kwargs)
    assert res.success
    return res, precise, loose, branch_id


def test_near_critical_rows_leave_worst_residuals() -> None:
    res, precise, loose, branch_id = _solve(quality_summary_top_n=10)
    ranked = [row.measurement_id for row in res.worst_residuals]
    listed = {row.measurement_id: row for row in res.near_critical_measurements}

    assert precise not in ranked
    assert ranked[0] == loose  # its partner stays in the ranking
    assert not set(ranked) & set(listed)

    row = listed[precise]
    assert row.kind == "P"
    assert 0.0 < row.omega_ratio < 1e-4
    assert np.isfinite(row.normalized_residual) and row.normalized_residual > 4.0
    assert not row.is_pseudo
    assert (row.object_kind, row.object_id, row.branch_side) == (OBJ_BRANCH, branch_id, 0)
    ratios = [r.omega_ratio for r in res.near_critical_measurements]
    assert ratios == sorted(ratios)  # least reliable first


def test_zero_threshold_restores_the_plain_ranking() -> None:
    plain, precise, loose, _ = _solve(quality_summary_near_critical=0.0, quality_summary_top_n=20)
    res, *_ = _solve(quality_summary_top_n=10)

    assert plain.near_critical_measurements == []
    plain_ids = [row.measurement_id for row in plain.worst_residuals]
    assert set(plain_ids[:2]) == {precise, loose}
    rns = [row.normalized_residual for row in plain.worst_residuals]
    assert rns == sorted(rns, reverse=True)
    # The default ranking is the plain one without the near-critical rows.
    listed = {row.measurement_id for row in res.near_critical_measurements}
    assert [row.measurement_id for row in res.worst_residuals] == [
        i for i in plain_ids if i not in listed
    ][:10]


def test_report_statistics_skip_near_critical_rows() -> None:
    model, precise, loose, _ = _case14_with_near_critical_flow()
    assert estimate(model, algorithm="wls", include_quality_summary=False).success

    plain = compute_normalized_residuals_report(model, top_n=3, near_critical_threshold=0.0)
    report = compute_normalized_residuals_report(model, top_n=3)

    # The rn array itself is the same; only the ranking statistics differ.
    np.testing.assert_array_equal(report.rn, plain.rn)
    assert plain.near_critical_meas_ids == []
    assert precise in [rec.meas_id for rec in plain.top_records]
    finite = np.isfinite(plain.rn)
    assert plain.rn_max_finite == float(np.max(plain.rn[finite]))
    assert plain.n_above_threshold == int(np.sum(plain.rn[finite] > 3.0))

    assert precise in report.near_critical_meas_ids
    assert precise not in [rec.meas_id for rec in report.top_records]
    assert report.top_records[0].meas_id == loose
    ranked = finite & ~np.isin(report.meas_ids, report.near_critical_meas_ids)
    assert report.rn_max_finite == float(np.max(report.rn[ranked]))
    assert report.n_above_threshold == int(np.sum(report.rn[ranked] > 3.0))
    assert report.n_above_threshold == plain.n_above_threshold - 1
    ratios = report.omega_ratio[np.isin(report.meas_ids, report.near_critical_meas_ids)]
    assert np.all(ratios < 1e-4)
    order = [int(np.where(report.meas_ids == i)[0][0]) for i in report.near_critical_meas_ids]
    assert list(report.omega_ratio[order]) == sorted(report.omega_ratio[order])


def test_top_worst_residuals_complements_the_near_critical_list() -> None:
    # Row 0 is nearly critical (a precise reading beside a very loose one),
    # rows 2-3 are redundant, row 4 is critical (r_N = inf).
    H = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    sigma2 = np.array([1.0, 1e6, 1.0, 1.0, 1.0])
    r = np.array([0.1, 1.0, 0.2, -0.3, 0.0])
    z = np.ones(5)
    index = MeasurementIndex(
        kind=np.full(5, KIND_VOLTAGE, dtype=np.int64),
        object_kind=np.zeros(5, dtype=np.int64),
        object_pos=np.zeros(5, dtype=np.int64),
        branch_side=np.full(5, -1, dtype=np.int64),
        meas_id=np.arange(101, 106, dtype=np.int64),
    )
    normalized = nr.compute_normalized_residuals(r, H, sigma2)

    def ranked(threshold: float) -> list[int]:
        rows = top_worst_residuals(
            r, sigma2, H, index, z, n=10, normalized=normalized, near_critical=threshold
        )
        return [row.measurement_id for row in rows]

    finite = np.isfinite(normalized.rn)
    order = np.argsort(-np.where(finite, normalized.rn, -np.inf))
    assert ranked(0.0) == [int(index.meas_id[i]) for i in order if finite[i]]
    listed = [row.measurement_id for row in near_critical_measurements(normalized, index)]
    assert listed == [105, 101]
    assert ranked(nr.NEAR_CRITICAL_THRESHOLD) == [i for i in ranked(0.0) if i not in listed]
