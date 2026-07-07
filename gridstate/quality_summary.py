"""Сводка качества SE для ``SEResult``.

После сходимости ``estimate()`` собирает:

* ``Chi2Summary`` — ``J = Σ(r²/σ²)`` + df + порог χ² + ``passes``;
* ``worst_residuals`` — top-N измерений по ``|r_N| = |r|/√diag(Ω)``;
* ``worst_imbalance`` — top-N узлов по ``|imbalance_p|`` из ``model.nodes``;
* ``observability_warnings`` — ID узлов с нулевыми столбцами ``H``.

Используется ровно один проход по ``H``/``R⁻¹``: ``Ω = R − H G⁻¹ Hᵀ``
считается через Cholesky(G) + блочную треугольную подстановку по строкам
``H`` — прежняя плотная алгебра (``H.toarray()`` + ``solve(G, Hᵀ)``) на
крупных моделях (десятки тысяч мер) стоила ~40% всего времени SE. Логика
повторяет
``gridstate.validation.bad_data._normalized_residuals`` — здесь вынесено
локально, чтобы не тянуть зависимость на ``estimate()`` напрямую и оставить
summary дёшевой опцией (можно отключить через
``include_quality_summary=False``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import chi2 as _chi2_dist

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


def _normalized_residuals(
    r: np.ndarray,
    H: csr_matrix,
    sigma2: np.ndarray,
    rows_mask: np.ndarray | None = None,
) -> np.ndarray:
    """``r_N = |r| / √diag(Ω)``, где ``Ω = R − H G⁻¹ Hᵀ``.

    Если ``Ω_ii`` численно ≤ 0 — non-redundant измерение, возвращаем ``inf``.
    ``diag(H G⁻¹ Hᵀ)`` собирается через Cholesky(G) + блочную треугольную
    подстановку по строкам ``H`` — без материализации плотной ``H`` (m×n)
    и без полного ``solve(G, Hᵀ)``, которые на крупных моделях доминировали
    во времени всего SE.

    Args:
        rows_mask: (m,) bool — считать ``r_N`` только для этих строк
            (остальные получают ``nan``). ``G`` всегда собирается по ВСЕМ
            строкам ``H`` (веса всех измерений участвуют в оценке) —
            маскируется только дорогая блочная подстановка. ``None`` — все.
    """
    if r.size == 0:
        return np.array([], dtype=np.float64)

    from scipy.linalg import cho_factor, solve_triangular
    from scipy.sparse import diags

    H_csr = H.tocsr()
    R_inv_diag = 1.0 / sigma2
    # G = Hᵀ · R⁻¹ · H — sparse-сборка (дёшево), факторизация — плотный
    # Cholesky: G SPD, и diag(H G⁻¹ Hᵀ) = ‖L⁻¹·hᵢ‖² требует лишь ОДНОЙ
    # треугольной подстановки (BLAS trsm, многопоточно) вместо полного
    # solve. Sparse-альтернатива (splu + блочный solve) здесь медленнее:
    # SuperLU слабо векторизован по множественным правым частям; cvxopt-CHOLMOD
    # (полный и половинный sys=L solve) — однопоточный, проигрывает BLAS-trsm
    # (замер 2026-07-07: 3.6-6.1с против 2.5с на Юге).
    G = np.asarray((H_csr.T @ diags(R_inv_diag) @ H_csr).todense())
    try:
        L, lower = cho_factor(G, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        logger.warning("quality_summary: G не инвертируется — r_N = inf (%s)", exc)
        return np.full_like(r, np.inf, dtype=np.float64)

    m = H_csr.shape[0]
    row_idx = np.arange(m) if rows_mask is None else np.where(rows_mask)[0]
    HGH_diag = np.full(m, np.nan, dtype=np.float64)
    # Блоками по строкам H — ограничивает память под dense RHS (n_state × block).
    block = 4096
    for s in range(0, row_idx.size, block):
        sel = row_idx[s : s + block]
        Y = solve_triangular(
            L, H_csr[sel, :].toarray().T, lower=lower, check_finite=False
        )  # (n_state × b) = L⁻¹ · Hbᵀ
        HGH_diag[sel] = np.einsum("ij,ij->j", Y, Y)

    omega_diag = sigma2 - HGH_diag
    omega_diag = np.where(omega_diag > 1e-12, omega_diag, np.nan)
    rn = np.abs(r) / np.sqrt(omega_diag)
    if rows_mask is not None:
        # немаскированные строки — nan (не участвуют в топе), маскированные
        # с Ω≤0 — inf (non-redundant, прежняя семантика).
        bad = np.isnan(rn) & (np.asarray(rows_mask, dtype=bool))
        rn = np.where(bad, np.inf, rn)
        return rn
    return np.where(np.isnan(rn), np.inf, rn)


def compute_chi2(
    r: np.ndarray,
    sigma2: np.ndarray,
    n_state: int,
    alpha: float = 0.05,
) -> Chi2Summary:
    """Посчитать ``J = Σ(r²/σ²)`` и сравнить с χ²(df, 1−α)."""
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
    """Топ-``n`` измерений по ``|r_N|`` (десятками; default 10).

    ``r`` и ``z`` берутся в ``p.u.`` (как build_z_and_r их и возвращает) —
    исходные единицы измерений недоступны из ``MeasurementIndex``, поэтому
    в ``ResidualRow.value/expected/residual`` записываются p.u.-значения.
    Для исходных единиц следует смотреть ``model.measurements[id].value``
    и ``estimated_si`` (заполнены ``write_measurement_estimates``).

    Args:
        network_pu: если задан — ``object_pos`` из ``meas_index``
            разрешается в ``id`` объекта (``bus_ids``/``branch_ids``) для
            ``ResidualRow.object_id``.
        is_pseudo: (m,) bool в z-порядке — пометка псевдо-приоров.
        scope: ``"real"`` (default) — в топ идут только реальные измерения
            (телеметрия); псевдо-приоры (инжекц-prior, pseudo-V) исключены —
            иначе они вытесняют телеметрию (на крупных моделях 8/10 топа —
            наши же приоры с r_N до сотен). Заодно дешевле: блочная
            подстановка идёт только по real-строкам (m падает в ~2.5×).
            ``"all"`` — прежняя семантика (все строки z-вектора).
            Без ``is_pseudo`` scope="real" эквивалентен "all".
    """
    if r.size == 0:
        return []
    rows_mask = None
    if scope == "real" and is_pseudo is not None:
        rows_mask = ~np.asarray(is_pseudo, dtype=bool)
        if not rows_mask.any():
            return []
    rn = _normalized_residuals(r, H, sigma2, rows_mask=rows_mask)
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
            if obj_kind == 0 and 0 <= obj_pos < network_pu.n_bus:
                obj_id = int(network_pu.bus_ids[obj_pos])
            elif obj_kind == 1 and 0 <= obj_pos < network_pu.n_branch:
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
    """Топ-``n`` узлов по ``|imbalance_p|`` после ``write_results_to_model``.

    ``imbalance_p/q`` уже посчитаны в МВт/МВАр (см. ``gridstate/units.py::
    write_results_to_model``). Берутся только узлы с ``status=True``.
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
    """ID узлов, чьи столбцы в ``H`` пусты — недонаблюдаемые δ или V.

    Аналог ``gridstate.validation.observability`` но дёшево: реиспользуем
    уже посчитанную ``H``, не пересобираем её для flat-старта.

    Раскладка столбцов ``H`` (WLS) — ``[δ_non_slack, V]`` длиной
    ``2·n_bus − 1``. Для IPM-расширенного state дополнительные столбцы
    (Pgen/Qgen/Pnag/Qnag) игнорируются — это box-vars, не узлы.
    """
    if H.shape[0] == 0 or H.shape[1] == 0:
        return []
    if hasattr(H, "multiply"):
        # sparse: нормы столбцов без материализации плотной (m×n)
        col_norms = np.sqrt(np.asarray(H.multiply(H).sum(axis=0)).ravel())
    else:
        col_norms = np.linalg.norm(np.asarray(H), axis=0)
    n_state_wls = 2 * n_bus - 1
    # Ограничиваем анализ первыми n_state_wls столбцами — остальные
    # (если есть) это IPM box-vars и не имеют 1-к-1 mapping на узел.
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
