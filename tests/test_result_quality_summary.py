"""Юнит-тесты для quality summary в ``SEResult``.

Покрывает:
    * заполнение полей ``chi2``/``worst_residuals``/``worst_imbalance``/
      ``observability_warnings`` после ``estimate()``;
    * сортировку ``worst_residuals`` по убыванию ``|r_N|``;
    * сходимость ``chi2.value > 0`` и ``dof = m − n_state``;
    * backward-compat: ``include_quality_summary=False`` оставляет defaults.

Используется та же 3-узловая сеть, что и в ``tests/test_bad_data.py``,
без шума на чистых синтетических измерениях.
"""

from __future__ import annotations

import numpy as np

from gridstate.api import estimate
from gridstate.result import (
    Chi2Summary,
    ImbalanceRow,
    ResidualRow,
    SEResult,
)
from tests.test_bad_data import _three_bus_with_clean_measurements


def test_seresult_quality_summary_populated() -> None:
    """После ``estimate()`` chi2/worst_*/observability должны быть заполнены."""
    m, _, _ = _three_bus_with_clean_measurements()
    res = estimate(m, tolerance=1e-10)

    assert res.success
    # chi2
    assert isinstance(res.chi2, Chi2Summary)
    assert res.chi2.dof > 0
    assert res.chi2.value >= 0.0
    # На чистых данных без шума J ≈ 0 → должен пройти порог.
    assert res.chi2.passes
    # threshold не nan при dof>0
    assert np.isfinite(res.chi2.threshold)

    # worst_residuals
    assert isinstance(res.worst_residuals, list)
    assert len(res.worst_residuals) > 0
    assert all(isinstance(row, ResidualRow) for row in res.worst_residuals)

    # отсортированы по убыванию |r_N|
    rns = [abs(row.normalized_residual) for row in res.worst_residuals]
    assert rns == sorted(rns, reverse=True)

    # каждый row имеет валидный label
    valid_kinds = {"V", "P", "Q", "I", "P_inj", "Q_inj", "?"}
    for row in res.worst_residuals:
        assert row.kind in valid_kinds or row.kind.endswith("_pr") or row.kind.endswith("bal")
        assert row.measurement_id > 0
        assert np.isfinite(row.value)
        assert np.isfinite(row.expected)
        assert np.isfinite(row.residual)
        # На чистых данных r ≈ value − expected
        assert abs((row.value - row.expected) - row.residual) < 1e-6

    # worst_imbalance
    assert isinstance(res.worst_imbalance, list)
    assert all(isinstance(row, ImbalanceRow) for row in res.worst_imbalance)
    # 3-узловая сеть: топ-10 == 3 узла
    assert len(res.worst_imbalance) == 3
    # отсортированы по |imbalance_p|
    imbs = [abs(row.imbalance_p_mw) for row in res.worst_imbalance]
    assert imbs == sorted(imbs, reverse=True)

    # observability_warnings — пуст для полностью наблюдаемой сети
    assert isinstance(res.observability_warnings, list)
    assert res.observability_warnings == []


def test_seresult_chi2_dof_matches_m_minus_n() -> None:
    """``chi2.dof == n_measurements − n_state_vars`` (n_state = 2n_bus − 1)."""
    m, _, _ = _three_bus_with_clean_measurements()
    res = estimate(m, tolerance=1e-10)
    assert res.chi2 is not None

    n_meas = sum(1 for me in m.measurements if me.status)
    n_state = 2 * 3 - 1  # 3 узла → state size = 5
    assert res.chi2.dof == n_meas - n_state


def test_seresult_quality_summary_disabled() -> None:
    """``include_quality_summary=False`` оставляет defaults."""
    m, _, _ = _three_bus_with_clean_measurements()
    res = estimate(m, tolerance=1e-10, include_quality_summary=False)

    assert res.success
    # default-empty / None
    assert res.chi2 is None
    assert res.worst_residuals == []
    assert res.worst_imbalance == []
    assert res.observability_warnings == []


def test_seresult_backward_compat_default_construction() -> None:
    """``SEResult(model=...)`` без новых полей продолжает работать."""

    # Минимальная mock-модель — нужна только ссылка для dataclass.
    class _Stub:
        pass

    stub = _Stub()
    res = SEResult(model=stub)  # type: ignore[arg-type]
    assert res.success is False
    assert res.iterations == 0
    # Все новые поля имеют backward-compat defaults.
    assert res.chi2 is None
    assert res.worst_residuals == []
    assert res.worst_imbalance == []
    assert res.observability_warnings == []


def test_seresult_worst_residuals_top_n_kwarg() -> None:
    """``quality_summary_top_n`` ограничивает размер топа."""
    m, _, _ = _three_bus_with_clean_measurements()
    res = estimate(m, tolerance=1e-10, quality_summary_top_n=3)
    assert res.success
    assert len(res.worst_residuals) <= 3
    assert len(res.worst_imbalance) <= 3


def test_seresult_worst_residuals_object_binding() -> None:
    """``ResidualRow`` несёт объектную привязку: object_kind/object_id/σ/is_pseudo."""
    m, _, _ = _three_bus_with_clean_measurements()
    res = estimate(m, tolerance=1e-10)
    assert res.success
    assert len(res.worst_residuals) > 0

    node_ids = {int(n.id) for n in m.nodes}
    branch_ids = {int(b.id) for b in m.branches}
    for row in res.worst_residuals:
        assert row.object_kind in (0, 1)
        if row.object_kind == 0:
            assert row.object_id in node_ids
            assert row.branch_side == -1
        else:
            assert row.object_id in branch_ids
            assert row.branch_side in (0, 1)
        assert np.isfinite(row.sigma) and row.sigma > 0
        # Синтетические измерения — не псевдо.
        assert row.is_pseudo is False


def _with_pseudo_outlier():
    """Clean 3-bus model plus one pseudo V-prior with a gross deviation."""
    from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE

    m, _, _ = _three_bus_with_clean_measurements()
    m.measurements.add(
        {
            "id": 900,
            "object_type": int(OBJ_NODE),
            "object_id": 2,
            "measurement_type": int(KIND_VOLTAGE),
            "value": 90.0,  # ~18 kV off the true state — dominates any top
            "variance": 0.01,
            "status": True,
            "quality": 0,
            "branch_side": -1,
            "is_pseudo": True,
        }
    )
    return m


def test_worst_residuals_scope_real_excludes_pseudo() -> None:
    """Default scope="real": pseudo rows never enter worst_residuals."""
    m = _with_pseudo_outlier()
    res = estimate(m, tolerance=1e-10)
    assert len(res.worst_residuals) > 0
    assert all(row.is_pseudo is False for row in res.worst_residuals)
    assert all(row.measurement_id != 900 for row in res.worst_residuals)


def test_worst_residuals_scope_all_keeps_pseudo() -> None:
    """scope="all" restores the legacy semantics: the gross pseudo tops the list."""
    m = _with_pseudo_outlier()
    res = estimate(m, tolerance=1e-10, quality_summary_scope="all")
    assert len(res.worst_residuals) > 0
    top = res.worst_residuals[0]
    assert top.measurement_id == 900
    assert top.is_pseudo is True
