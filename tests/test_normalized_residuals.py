"""The bad-data test uses the shared sparse ``r_N`` path (no ``m × m`` algebra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.contract.serialize import load_se_input_npz
from gridstate.validation._diagnostics import compute_diagnostics
from gridstate.validation.bad_data import _normalized_residuals


_DATA_DIR = Path(__file__).parent / "test_data" / "ieee"


def _dense_reference(diag) -> np.ndarray:
    """The former dense formula: ``Ω = R − H (Hᵀ R⁻¹ H)⁻¹ Hᵀ``."""
    H = diag.H.toarray()
    G = H.T @ np.diag(1.0 / diag.sigma2) @ H
    hgh = np.einsum("ij,ji->i", H, np.linalg.solve(G, H.T))
    omega = diag.sigma2 - hgh
    omega = np.where(omega > 1e-12, omega, np.nan)
    rn = np.abs(diag.r) / np.sqrt(omega)
    return np.where(np.isnan(rn), np.inf, rn)


def _solved_diagnostics(case: str):
    model = load_se_input_npz(_DATA_DIR / f"{case}.npz").model
    assert estimate(model, algorithm="wls", include_quality_summary=False).success
    return compute_diagnostics(model, model.measurements)


@pytest.mark.parametrize("case", ["case14", "case30", "case118"])
def test_bad_data_rn_matches_dense_reference(case: str) -> None:
    diag = _solved_diagnostics(case)
    rn = _normalized_residuals(diag)
    ref = _dense_reference(diag)
    finite = np.isfinite(ref)
    assert finite.sum() > 0.9 * ref.size
    np.testing.assert_array_equal(np.isfinite(rn), finite)
    np.testing.assert_allclose(rn[finite], ref[finite], rtol=1e-12, atol=1e-12)


class _NoDense:
    """Stand-in for ``R⁻¹`` that refuses to be densified."""

    def toarray(self) -> np.ndarray:
        raise AssertionError("dense m x m R^-1 must not be formed")


def test_bad_data_rn_does_not_densify_r_inv() -> None:
    diag = _solved_diagnostics("case14")
    expected = _dense_reference(diag)
    rn = _normalized_residuals(diag._replace(R_inv=_NoDense()))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(rn[finite], expected[finite], rtol=1e-12, atol=1e-12)
