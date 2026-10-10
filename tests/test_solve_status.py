"""Solver-independent termination status (``SolveStatus``) and ``success_policy``.

* every exit of :func:`solve_wls` / :func:`solve_ipm` maps to one status;
* forced cases: a tiny iteration budget, the WLS trust-region collapse with
  gross errors on IEEE case14 (issue #57), the IPM schedule completed with an
  exhausted final inner solve;
* ``estimate(success_policy=...)``: ``"legacy"`` leaves ``success`` alone,
  ``"status"`` derives it from the status; ``objective_wls`` is comparable
  between the solvers;
* the pipeline option: validation, manifest, event stats, and the re-solve
  revert (#44) acting on IPM re-solves under ``"status"``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import gridstate.pipeline.passes as passes_mod
from gridstate import SolveStatus, estimate
from gridstate.algorithms.ipm import solve_ipm
from gridstate.algorithms.status import CONVERGED_STATUSES, success_from_status
from gridstate.algorithms.wls import solve_wls
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.pipeline import PipelineConfig, manifest
from gridstate.state import StateLayout, flat_start
from gridstate.units import model_to_pu
from gridstate.working import Working
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r
from tests._golden import load_ieee_input


_IEEE_DIR = Path(__file__).parent / "test_data" / "ieee"


# ---------------------------------------------------------------------------
# The enum and the uniform rule
# ---------------------------------------------------------------------------


def test_status_values_are_stable_codes() -> None:
    assert [s.value for s in SolveStatus] == [
        "kkt",
        "mu_schedule_done",
        "tr_collapsed",
        "max_iter",
        "stalled",
        "diverged",
        "error",
    ]
    assert str(SolveStatus.TR_COLLAPSED) == "tr_collapsed"
    assert SolveStatus("kkt") is SolveStatus.KKT
    assert SolveStatus.KKT == "kkt"  # str-valued


def test_success_from_status() -> None:
    assert {SolveStatus.KKT, SolveStatus.MU_SCHEDULE_DONE} == CONVERGED_STATUSES
    for s in SolveStatus:
        assert success_from_status(s) is (s in CONVERGED_STATUSES)
    assert success_from_status("mu_schedule_done") is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WLS
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case14() -> dict[str, Any]:
    model = load_se_input_npz(_IEEE_DIR / "case14.npz").model
    pu = model_to_pu(model)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)
    z, r_matrix, idx = build_z_and_r(model, model.measurements, pu)
    return {
        "model": model,
        "pu": pu,
        "layout": layout,
        "Y": (ybus, yf, yt),
        "z": z,
        "R": r_matrix,
        "idx": idx,
    }


def _wls(p: dict[str, Any], z: np.ndarray, **kw: Any) -> Any:
    ybus, yf, yt = p["Y"]
    return solve_wls(
        flat_start(p["layout"]), z, p["R"], ybus, yf, yt, p["idx"], p["layout"], p["pu"], **kw
    )


def _gross(z: np.ndarray, seed: int) -> np.ndarray:
    """5..20 p.u. gross errors on a seeded 10 % of the rows (as in test_wls_robustness)."""
    rng = np.random.default_rng(seed)
    zz = z.copy()
    k = rng.choice(len(z), size=max(1, len(z) // 10), replace=False)
    zz[k] += rng.choice([-1, 1], size=k.size) * rng.uniform(5.0, 20.0, size=k.size)
    return zz


def test_wls_clean_is_kkt(case14) -> None:
    res = _wls(case14, case14["z"])
    assert res.success
    assert res.solve_status is SolveStatus.KKT


def test_wls_iteration_budget_is_max_iter(case14) -> None:
    res = _wls(case14, case14["z"], max_iterations=1)
    assert not res.success
    assert res.iterations == 1
    assert res.solve_status is SolveStatus.MAX_ITER


def test_wls_trust_region_collapse_with_gross_errors(case14) -> None:
    """Issue #57: a chain of rejections drives the radius below the tolerance and
    the J-increasing tiny step is force-accepted. Legacy success stays True,
    the status tells it apart from a stationary point."""
    z = _gross(case14["z"], seed=0)
    res = _wls(case14, z, tolerance=1e-6, max_iterations=50)
    assert res.success  # legacy flag unchanged
    assert res.solve_status is SolveStatus.TR_COLLAPSED
    before = _wls(case14, z, tolerance=1e-6, max_iterations=res.iterations - 1)
    assert res.objective > before.objective


def test_wls_max_step_below_tolerance_is_tr_collapsed(case14) -> None:
    """The first step is clamped below the tolerance and force-accepted."""
    res = _wls(case14, case14["z"], tolerance=1e-6, max_step=1e-7)
    assert res.success and res.iterations == 1
    assert res.solve_status is SolveStatus.TR_COLLAPSED


def test_wls_lambda_limit_is_stalled(case14) -> None:
    res = _wls(case14, _gross(case14["z"], seed=0), tolerance=1e-9, max_iterations=50)
    assert not res.success
    assert res.iterations < 50
    assert res.solve_status is SolveStatus.STALLED


def test_wls_empty_measurements_is_error(case14) -> None:
    p = case14
    ybus, yf, yt = p["Y"]
    z, r_matrix, idx = build_z_and_r(p["model"], Working.empty().measurements, p["pu"])
    assert z.shape == (0,)
    res = solve_wls(flat_start(p["layout"]), z, r_matrix, ybus, yf, yt, idx, p["layout"], p["pu"])
    assert not res.success
    assert res.solve_status is SolveStatus.ERROR


def test_wls_status_diagnostic_does_not_change_the_iterate(case14) -> None:
    """The extra undamped solve behind the status check never feeds back: the
    collapsed run is bit-identical to a run capped at the same iteration count."""
    z = _gross(case14["z"], seed=0)
    res = _wls(case14, z, tolerance=1e-6, max_iterations=50)
    capped = _wls(case14, z, tolerance=1e-6, max_iterations=res.iterations)
    np.testing.assert_array_equal(res.x, capped.x)
    assert res.objective == capped.objective


# ---------------------------------------------------------------------------
# IPM (scalar problems)
# ---------------------------------------------------------------------------


def _scalar(z_value: float, *, jac_sign: float = 1.0, r_inv: float = 1.0) -> dict[str, Any]:
    z = np.array([z_value])
    return {
        "residual_fn": lambda x: z - np.array([x[0]]),
        "jacobian_fn": lambda x: csr_matrix(np.array([[jac_sign]])),
        "r_inv_diag": np.array([r_inv]),
    }


def _box(lo: float, hi: float) -> dict[str, Any]:
    return {"box_idx": np.array([0]), "box_lo": np.array([lo]), "box_hi": np.array([hi])}


def test_ipm_clean_is_kkt() -> None:
    res = solve_ipm(np.array([0.5]), **_scalar(3.0), **_box(0.0, 10.0))
    assert res.status == "kkt"
    assert res.solve_status is SolveStatus.KKT


def test_ipm_schedule_done_with_loose_gradient() -> None:
    """Large weights: the inner Newton converges by step at μ_min while |grad|∞
    stays above the strict KKT threshold — legacy ``completed``."""
    res = solve_ipm(np.array([1.0]), **_scalar(10.0, r_inv=1e8), **_box(0.0, 2.0))
    assert res.status == "completed"
    assert res.solve_status is SolveStatus.MU_SCHEDULE_DONE
    assert res.success


def test_ipm_exhausted_final_inner_is_max_iter() -> None:
    """``inner_max=1``: the schedule reaches μ_min but the final inner solve never
    converges. Legacy reports ``completed`` / success."""
    res = solve_ipm(
        np.array([1.0]), **_scalar(10.0), **_box(0.0, 2.0), inner_tol=1e-12, inner_max=1
    )
    assert res.status == "completed" and res.success
    assert res.solve_status is SolveStatus.MAX_ITER


def test_ipm_exhausted_outer_budget_is_max_iter() -> None:
    res = solve_ipm(np.array([1.0]), **_scalar(10.0, r_inv=1e8), **_box(0.0, 2.0), outer_max=2)
    assert res.status == "stalled"
    assert res.solve_status is SolveStatus.MAX_ITER


def test_ipm_armijo_failure_is_stalled() -> None:
    res = solve_ipm(np.array([1.0]), **_scalar(10.0, jac_sign=-1.0), **_box(0.0, 100.0))
    assert res.status == "stalled"
    assert res.solve_status is SolveStatus.STALLED


def test_ipm_non_finite_step_is_error() -> None:
    p = _scalar(3.0)
    p["jacobian_fn"] = lambda x: csr_matrix(np.array([[np.nan]]))
    res = solve_ipm(np.array([0.5]), **p, **_box(0.0, 10.0))
    assert res.status == "error"
    assert res.solve_status is SolveStatus.ERROR


def test_ipm_without_boxes() -> None:
    res = solve_ipm(np.array([0.5]), **_scalar(3.0))
    assert res.solve_status is SolveStatus.KKT
    # Nonlinear h = x³ with large weights: the Newton step drops below inner_tol
    # while |grad|∞ is still above the KKT threshold.
    z = np.array([8.0])
    weighted = solve_ipm(
        np.array([3.0]),
        residual_fn=lambda x: z - x[:1] ** 3,
        jacobian_fn=lambda x: csr_matrix(np.array([[3.0 * x[0] ** 2]])),
        r_inv_diag=np.array([1e8]),
    )
    assert weighted.status == "stalled"  # legacy: no schedule, gradient above threshold
    assert not weighted.success
    assert weighted.solve_status is SolveStatus.MU_SCHEDULE_DONE
    assert weighted.x[0] == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# estimate(): success_policy and objective_wls
# ---------------------------------------------------------------------------


def _estimate(name: str, **kw: Any) -> Any:
    return estimate(load_ieee_input(name).model, include_quality_summary=False, **kw)


@pytest.mark.parametrize(
    ("algorithm", "status"), [("wls", SolveStatus.KKT), ("ipm", SolveStatus.MU_SCHEDULE_DONE)]
)
def test_estimate_reports_status_and_policy_is_neutral_on_clean_data(algorithm, status) -> None:
    legacy = _estimate("case14", algorithm=algorithm)
    by_status = _estimate("case14", algorithm=algorithm, success_policy="status")
    assert legacy.solve_status is status
    assert legacy.success and by_status.success
    np.testing.assert_array_equal(legacy.v_pu, by_status.v_pu)
    np.testing.assert_array_equal(legacy.delta_rad, by_status.delta_rad)
    assert (legacy.iterations, legacy.objective_value, legacy.objective_wls) == (
        by_status.iterations,
        by_status.objective_value,
        by_status.objective_wls,
    )
    assert legacy.convergence_status == by_status.convergence_status


def test_estimate_ipm_max_iter_fails_under_status_policy() -> None:
    legacy = _estimate("case118_degraded", algorithm="ipm", max_iterations=1)
    assert legacy.convergence_status == "completed" and legacy.success
    assert legacy.solve_status is SolveStatus.MAX_ITER
    by_status = _estimate(
        "case118_degraded", algorithm="ipm", max_iterations=1, success_policy="status"
    )
    assert not by_status.success
    assert by_status.message.startswith("solve status max_iter: ")
    np.testing.assert_array_equal(legacy.v_pu, by_status.v_pu)


def test_estimate_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="success_policy"):
        _estimate("case14", success_policy="strict")


def test_objective_wls_is_comparable_between_solvers() -> None:
    """``objective_value`` differs by definition (½, augmented rows, Huber weights);
    ``objective_wls`` is the same function for both solvers: the plain WLS J,
    so the WLS minimiser cannot be beaten by the IPM point."""
    wls = _estimate("case14", algorithm="wls", huber_c=0.0)
    ipm = _estimate("case14", algorithm="ipm", huber_c=0.0)
    assert wls.objective_wls == pytest.approx(wls.objective_value, rel=1e-12)
    # The IPM internal objective carries the ½ convention: about half of J.
    assert ipm.objective_value == pytest.approx(0.5 * ipm.objective_wls, rel=0.05)
    assert ipm.objective_wls >= wls.objective_wls * (1.0 - 1e-9)
    assert ipm.objective_wls == pytest.approx(wls.objective_wls, rel=0.05)


# ---------------------------------------------------------------------------
# Pipeline option
# ---------------------------------------------------------------------------


def test_pipeline_option_default_validation_and_manifest() -> None:
    assert PipelineConfig().success_policy == "legacy"
    with pytest.raises(ValueError, match="success_policy"):
        PipelineConfig(success_policy="strict")
    fields = {f["name"]: f for f in manifest()["params"]}
    md = fields["success_policy"]
    assert md["choices"] == ["legacy", "status"]
    assert md["default"] == "legacy"


def _run(name: str, cfg: PipelineConfig) -> tuple[Any, dict[str, Any]]:
    events: list[dict] = []
    out = run_se(load_ieee_input(name), config=cfg, on_event=events.append, validate=False)
    return out, {e["name"]: e["stats"] for e in events if e["type"] == "step_done"}


def test_pipeline_events_carry_status() -> None:
    _, stats = _run("case14", PipelineConfig())
    assert stats["estimate"]["solve_status"] == "kkt"
    assert np.isfinite(stats["estimate"]["objective_wls"])


_BAD_DATA_ONLY: dict[str, Any] = {
    "bad_data": True,
    "bad_data_threshold": 2.0,
    "v_refine": False,
    "v_mirror": False,
    "v_mirror_chain": False,
    "refine_merge": False,
    "anti_overshoot": False,
}


@pytest.mark.parametrize("policy", ["legacy", "status"])
def test_ipm_resolve_revert_acts_under_status_policy(monkeypatch, policy) -> None:
    """#44 reverts a re-solve that lost convergence. A real IPM re-solve with a
    one-step inner budget ends in ``max_iter``: the legacy flag still says
    success (kept), the status policy reverts it."""
    real = passes_mod.estimate
    calls: list[int] = []

    def _estimate_capped(model: Any, **kw: Any) -> Any:
        calls.append(1)
        if len(calls) > 1:  # the re-solves
            kw["max_iterations"] = 1
        return real(model, **kw)

    monkeypatch.setattr(passes_mod, "estimate", _estimate_capped)
    cfg = replace(PipelineConfig(algorithm="ipm", success_policy=policy), **_BAD_DATA_ONLY)
    out, stats = _run("case14_degraded", cfg)
    st = stats["bad_data_repass"]
    assert len(calls) == 2
    assert st["solve_status"] == "max_iter"
    assert out.success
    if policy == "legacy":
        assert st["success"] is True
        assert "reverted" not in st
    else:
        assert st["success"] is False
        assert st["reverted"] is True
        monkeypatch.setattr(passes_mod, "estimate", real)
        ref, _ = _run("case14_degraded", replace(cfg, bad_data=False))
        np.testing.assert_array_equal(out.v_pu, ref.v_pu)
        np.testing.assert_array_equal(out.delta_rad, ref.delta_rad)
