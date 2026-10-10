"""Quality summary of the state estimate for ``SEResult``.

After ``estimate()`` converges it collects:

* ``Chi2Summary`` — ``J = Σ(r²/σ²)`` + df + χ² threshold + ``passes``;
* ``worst_residuals`` — top-N measurements by ``|r_N| = |r|/√diag(Ω)``;
* ``worst_imbalance`` — top-N nodes by ``|imbalance_p|`` from ``model.nodes``;
* ``observability_warnings`` — IDs of nodes with zero ``H`` columns.

Exactly one pass over ``H``/``R⁻¹`` is used: ``Ω = R − H G⁻¹ Hᵀ`` is
computed with sparse CHOLMOD and sparse right-hand sides (no cvxopt —
dense Cholesky + block triangular substitution). Dense algebra on large
models (tens of thousands of measurements) dominated the whole state
estimation time. The ``r_N`` computation lives in
:mod:`gridstate.algebra.normalized_residuals`, shared with the bad-data test
(``gridstate.validation.bad_data``). The summary can be disabled via
``include_quality_summary=False``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import chi2 as _chi2_dist

from gridstate.algebra.normalized_residuals import normalized_residuals
from gridstate.constants import MeasurementObjectType
from gridstate.result import (
    Chi2Summary,
    ImbalanceRow,
    ResidualRow,
    label_for_kind,
)


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working
    from gridstate.z_vector import MeasurementIndex


logger = logging.getLogger(__name__)


def compute_chi2(
    r: np.ndarray,
    sigma2: np.ndarray,
    n_state: int,
    alpha: float = 0.05,
) -> Chi2Summary:
    """Compute ``J = Σ(r²/σ²)`` and compare it with χ²(df, 1−α)."""
    m = int(r.shape[0])
    if m == 0:
        return Chi2Summary(value=0.0, dof=0, threshold=float("nan"), passes=True)
    safe_sigma2 = np.where(sigma2 > 1e-12, sigma2, 1e-12)
    j = float(np.sum((r * r) / safe_sigma2))
    df = m - n_state
    if df <= 0:
        return Chi2Summary(value=j, dof=df, threshold=float("nan"), passes=True)
    threshold = float(_chi2_dist.ppf(1.0 - alpha, df))
    return Chi2Summary(
        value=j,
        dof=df,
        threshold=threshold,
        passes=bool(j <= threshold),
    )


def top_worst_residuals(
    r: np.ndarray,
    sigma2: np.ndarray,
    H: csr_matrix,
    meas_index: MeasurementIndex,
    z: np.ndarray,
    *,
    n: int = 10,
    network_pu: NetworkPU | None = None,
    is_pseudo: np.ndarray | None = None,
    scope: str = "real",
) -> list[ResidualRow]:
    """Top-``n`` measurements by ``|r_N|`` (default 10).

    ``r`` and ``z`` are taken in ``p.u.`` (as build_z_and_r returns them) —
    the original measurement units are not available from ``MeasurementIndex``,
    so ``ResidualRow.value/expected/residual`` hold p.u. values.
    For the original units look at ``model.measurements[id].value``
    and ``estimated_si`` (filled by ``write_measurement_estimates``).

    Args:
        network_pu: if given, ``object_pos`` from ``meas_index`` is
            resolved to the object ``id`` (``bus_ids``/``branch_ids``) for
            ``ResidualRow.object_id``.
        is_pseudo: (m,) bool in z order — marks pseudo-priors.
        scope: ``"real"`` (default) — only real measurements (telemetry)
            enter the top; pseudo-priors (injection prior, pseudo-V) are
            excluded — otherwise they crowd out telemetry (on large models
            8 of the top 10 are our own priors with r_N up to hundreds).
            It is also cheaper: the block substitution runs over real rows
            only (m drops ~2.5x).
            ``"all"`` — the previous semantics (all z-vector rows).
            Without ``is_pseudo`` scope="real" is equivalent to "all".
    """
    if r.size == 0:
        return []
    rows_mask = None
    if scope == "real" and is_pseudo is not None:
        rows_mask = ~np.asarray(is_pseudo, dtype=bool)
        if not rows_mask.any():
            return []
    rn = normalized_residuals(r, H, sigma2, rows_mask=rows_mask)
    finite_mask = np.isfinite(rn)
    if not finite_mask.any():
        return []
    rn_sortable = np.where(finite_mask, rn, -np.inf)
    order = np.argsort(-rn_sortable)
    rows: list[ResidualRow] = []
    for pos in order[:n]:
        if not finite_mask[pos]:
            break
        kind_code = int(meas_index.kind[pos])
        z_val = float(z[pos])
        r_val = float(r[pos])
        h_val = z_val - r_val
        obj_kind = int(meas_index.object_kind[pos])
        obj_pos = int(meas_index.object_pos[pos])
        obj_id = 0
        if network_pu is not None:
            if obj_kind == MeasurementObjectType.NODE and 0 <= obj_pos < network_pu.n_bus:
                obj_id = int(network_pu.bus_ids[obj_pos])
            elif obj_kind == MeasurementObjectType.BRANCH and 0 <= obj_pos < network_pu.n_branch:
                obj_id = int(network_pu.branch_ids[obj_pos])
        rows.append(
            ResidualRow(
                measurement_id=int(meas_index.meas_id[pos]),
                kind=label_for_kind(kind_code),
                value=z_val,
                expected=h_val,
                residual=r_val,
                normalized_residual=float(rn[pos]),
                sigma=float(np.sqrt(sigma2[pos])),
                object_kind=obj_kind,
                object_id=obj_id,
                branch_side=int(meas_index.branch_side[pos]),
                is_pseudo=bool(is_pseudo[pos]) if is_pseudo is not None else False,
            )
        )
    return rows


def top_worst_imbalance(model: Working, n: int = 10) -> list[ImbalanceRow]:
    """Top-``n`` nodes by ``|imbalance_p|`` after ``write_results_to_model``.

    ``imbalance_p/q`` are already computed in MW/MVAr (see ``gridstate/units.py::
    write_results_to_model``). Only nodes with ``status=True`` are considered.
    """
    nodes_arr = model.nodes.to_numpy()
    if len(nodes_arr) == 0:
        return []
    active = nodes_arr[nodes_arr["status"]]
    if len(active) == 0:
        return []
    imb_p = active["imbalance_p"]
    if imb_p.size == 0:
        return []
    order = np.argsort(-np.abs(imb_p))
    rows: list[ImbalanceRow] = []
    for pos in order[:n]:
        row = active[pos]
        rows.append(
            ImbalanceRow(
                node_id=int(row["id"]),
                imbalance_p_mw=float(row["imbalance_p"]),
                imbalance_q_mvar=float(row["imbalance_q"]),
            )
        )
    return rows


def observability_warnings_from_H(
    H: csr_matrix,
    network_pu: NetworkPU,
    *,
    n_bus: int,
    non_slack_idx: np.ndarray,
    col_norm_tol: float = 1e-10,
) -> list[int]:
    """IDs of nodes whose ``H`` columns are empty — unobservable δ or V.

    An analogue of ``gridstate.validation.observability`` but cheap: reuse the
    already computed ``H`` instead of rebuilding it for a flat start.

    The ``H`` column layout (WLS) is ``[δ_non_slack, V]`` of length
    ``2·n_bus − 1``. For the IPM-extended state the additional columns
    (Pgen/Qgen/Pnag/Qnag) are ignored — they are box vars, not nodes.
    """
    if H.shape[0] == 0 or H.shape[1] == 0:
        return []
    if hasattr(H, "multiply"):
        # sparse: column norms without materialising a dense (m×n)
        col_norms = np.sqrt(np.asarray(H.multiply(H).sum(axis=0)).ravel())
    else:
        col_norms = np.linalg.norm(np.asarray(H), axis=0)
    n_state_wls = 2 * n_bus - 1
    # Restrict the analysis to the first n_state_wls columns — the rest
    # (if any) are IPM box vars and have no 1-to-1 mapping to a node.
    col_norms = col_norms[:n_state_wls]
    zero_cols = np.where(col_norms < col_norm_tol)[0]
    if zero_cols.size == 0:
        return []
    warnings: set[int] = set()
    for j in zero_cols:
        bus_pos = int(non_slack_idx[j]) if j < n_bus - 1 else int(j - (n_bus - 1))
        if 0 <= bus_pos < n_bus:
            warnings.add(int(network_pu.bus_ids[bus_pos]))
    return sorted(warnings)
