"""Robustness paths of the trust-region / Levenberg-Marquardt WLS solver.

``tests/test_wls.py`` covers the well-conditioned path (pure Gauss-Newton steps
inside the trust region). This module drives :func:`gridstate.algorithms.wls.
solve_wls` into the recovery machinery:

* the LM bisection of :func:`~gridstate.algorithms.wls._solve_in_trust_region`
  (``lam_floor > 0``), unit-tested on a hand-checkable 2x2 gain matrix;
* rejected steps followed by an accepted damped step (genuine recovery on IEEE
  case14 with gross measurement errors);
* the trust-region collapse (``lam > lam_max``) and the force-accept of a tiny
  step after a chain of rejections;
* the early exits (zero gain diagonal, non-finite objective, input checks).

The solver internals are observed through a transparent spy on
``_solve_in_trust_region`` (it records the arguments and the result and returns
the original result unchanged) instead of log text, so the assertions do not
depend on message wording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from gridstate.algorithms import wls
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.algorithms.status import SolveStatus
from gridstate.algorithms.wls import _solve_in_trust_region, solve_wls
from gridstate.constants import BranchType, NodeType
from gridstate.contract.serialize import load_se_input_npz
from gridstate.state import StateLayout, flat_start, pack
from gridstate.units import model_to_pu
from gridstate.working import Working
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
    KIND_CURRENT,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    build_z_and_r,
)


_IEEE_DIR = Path(__file__).parent / "test_data" / "ieee"


# ---------------------------------------------------------------------------
# _solve_in_trust_region on a 2x2 gain matrix
# ---------------------------------------------------------------------------

# Symmetric positive definite: eigenvalues ~4.08 and ~0.92.
_G = csc_matrix(np.array([[4.0, 0.5], [0.5, 1.0]]))
_RHS = np.array([2.0, 3.0])
_DIAG = _G.diagonal()


def _damped(lam: float) -> np.ndarray:
    """Reference solution of ``(G + lam*diag(G)) d = rhs`` (dense)."""
    a = _G.toarray() + lam * np.diag(_DIAG)
    return np.linalg.solve(a, _RHS)


class _ScriptedSolver:
    """KKT solver stub: delegates to scipy, but fails on the listed call numbers.

    ``fail_calls`` holds 1-based call numbers; ``mode`` selects how a failure
    looks to the caller (``"raise"`` or ``"nan"``). ``fail_after`` makes every
    call after that number fail.
    """

    def __init__(
        self,
        *,
        fail_calls: tuple[int, ...] = (),
        fail_after: int | None = None,
        mode: str = "raise",
    ) -> None:
        self._inner = KKTSolver("scipy")
        self.fail_calls = set(fail_calls)
        self.fail_after = fail_after
        self.mode = mode
        self.calls = 0

    def solve(self, G: csc_matrix, rhs: np.ndarray) -> np.ndarray:
        self.calls += 1
        failing = self.calls in self.fail_calls or (
            self.fail_after is not None and self.calls > self.fail_after
        )
        if failing:
            if self.mode == "raise":
                raise RuntimeError("scripted solver failure")
            return np.full_like(rhs, np.nan)
        return self._inner.solve(G, rhs)


def test_gn_mode_clamps_the_raw_step_to_the_radius() -> None:
    """``lam_floor == 0``: the Gauss-Newton direction is kept, its length is clamped."""
    d_gn = _damped(0.0)
    tr = 0.5
    assert np.max(np.abs(d_gn)) > tr
    d, step, lam_used, ok = _solve_in_trust_region(_G, _RHS, _DIAG, tr, 0.0, KKTSolver("scipy"))
    assert ok
    assert lam_used == 0.0
    assert step == tr
    np.testing.assert_allclose(d, d_gn * (tr / np.max(np.abs(d_gn))), rtol=1e-12)


def test_lm_mode_returns_the_floor_step_when_it_fits() -> None:
    """``lam_floor > 0`` and the damped step already inside the radius: no bisection."""
    lam = 0.1
    d, step, lam_used, ok = _solve_in_trust_region(_G, _RHS, _DIAG, 10.0, lam, KKTSolver("scipy"))
    assert ok
    assert lam_used == lam
    np.testing.assert_allclose(d, _damped(lam), rtol=1e-12)
    assert step == pytest.approx(np.max(np.abs(_damped(lam))), rel=1e-12)


def test_lm_bisection_lands_inside_the_radius_near_its_boundary() -> None:
    """Too long a floor step: lambda grows, then a log-bisection puts the step in
    ``[0.7*TR, TR]`` and the returned ``d`` solves the damped system for ``lam_used``."""
    tr = 0.1
    lam_floor = 1e-3
    assert np.max(np.abs(_damped(lam_floor))) > tr
    d, step, lam_used, ok = _solve_in_trust_region(
        _G, _RHS, _DIAG, tr, lam_floor, KKTSolver("scipy")
    )
    assert ok
    assert lam_used > lam_floor
    assert 0.7 * tr <= step <= tr
    assert step == pytest.approx(np.max(np.abs(d)), rel=1e-15)
    np.testing.assert_allclose(d, _damped(lam_used), rtol=1e-10)


def test_lm_bisection_skips_a_failed_midpoint_solve() -> None:
    """A failed (NaN) midpoint solve is treated as 'too long' and the bisection goes on."""
    tr = 0.1
    lam_floor = 1e-3
    # Call 1: floor step (too long). Call 2: the first lam_hi already fits.
    # Call 3: the first midpoint -> NaN.
    solver = _ScriptedSolver(fail_calls=(3,), mode="nan")
    d, step, lam_used, ok = _solve_in_trust_region(_G, _RHS, _DIAG, tr, lam_floor, solver)
    assert solver.calls > 3
    assert ok
    assert 0.0 < step <= tr
    np.testing.assert_allclose(d, _damped(lam_used), rtol=1e-10)


def test_lm_falls_back_to_scaling_when_no_lambda_solves() -> None:
    """Every solve after the floor step fails: the floor step is scaled to the radius."""
    tr = 0.1
    lam_floor = 1e-3
    solver = _ScriptedSolver(fail_after=1)
    d, step, lam_used, ok = _solve_in_trust_region(_G, _RHS, _DIAG, tr, lam_floor, solver)
    # 1 floor solve + 40 attempts to find lam_hi.
    assert solver.calls == 41
    assert ok
    assert lam_used == lam_floor
    assert step == tr
    d_lo = _damped(lam_floor)
    np.testing.assert_allclose(d, d_lo * (tr / np.max(np.abs(d_lo))), rtol=1e-12)


def test_lm_with_zero_diagonal_cannot_shrink_and_falls_back() -> None:
    """An all-zero ``diag_G`` makes the damping a no-op: no lambda shortens the step,
    so after 40 growth attempts the floor step is scaled to the radius."""
    tr = 0.1
    zero_diag = np.zeros(2)
    solver = _ScriptedSolver()
    d, step, _lam_used, ok = _solve_in_trust_region(_G, _RHS, zero_diag, tr, 1e-3, solver)
    assert solver.calls == 41
    assert ok
    assert step == tr
    d_gn = _damped(0.0)
    np.testing.assert_allclose(d, d_gn * (tr / np.max(np.abs(d_gn))), rtol=1e-12)


@pytest.mark.parametrize("lam_floor", [0.0, 1e-3])
def test_solver_exception_reports_failure(lam_floor: float) -> None:
    """A raising linear solver yields ``ok=False`` and a zero step, in both modes."""
    solver = _ScriptedSolver(fail_after=0)
    d, step, lam_used, ok = _solve_in_trust_region(_G, _RHS, _DIAG, 0.1, lam_floor, solver)
    assert not ok
    assert step == 0.0
    assert lam_used == lam_floor
    np.testing.assert_array_equal(d, np.zeros(2))


# ---------------------------------------------------------------------------
# solve_wls on IEEE case14 with gross errors
# ---------------------------------------------------------------------------


class _TrustRegionSpy:
    """Transparent wrapper of ``_solve_in_trust_region`` recording every call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, G, rhs, diag_G, tr_radius, lam_floor, solver):  # type: ignore[no-untyped-def]
        out = _solve_in_trust_region(G, rhs, diag_G, tr_radius, lam_floor, solver)
        self.calls.append(
            {"tr": tr_radius, "lam_floor": lam_floor, "lam_used": out[2], "step": out[1]}
        )
        return out


@pytest.fixture(scope="module")
def case14_problem():
    """Network, layout and clean ``z`` of the IEEE case14 fixture."""
    se_in = load_se_input_npz(_IEEE_DIR / "case14.npz")
    model = se_in.model
    pu = model_to_pu(model)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)
    z, r_matrix, idx = build_z_and_r(model, model.measurements, pu)
    return {
        "pu": pu,
        "layout": layout,
        "ybus": ybus,
        "yf": yf,
        "yt": yt,
        "z": z,
        "R": r_matrix,
        "idx": idx,
    }


def _with_gross_errors(z: np.ndarray, seed: int) -> np.ndarray:
    """Add 5..20 p.u. gross errors to a seeded 10 % of the measurements."""
    rng = np.random.default_rng(seed)
    zz = z.copy()
    k = rng.choice(len(z), size=max(1, len(z) // 10), replace=False)
    zz[k] += rng.choice([-1, 1], size=k.size) * rng.uniform(5.0, 20.0, size=k.size)
    return zz


def _run(problem, z: np.ndarray, **kwargs):  # type: ignore[no-untyped-def]
    p = problem
    return solve_wls(
        flat_start(p["layout"]),
        z,
        p["R"],
        p["ybus"],
        p["yf"],
        p["yt"],
        p["idx"],
        p["layout"],
        p["pu"],
        **kwargs,
    )


def test_rejected_step_recovers_through_the_damped_step(case14_problem, monkeypatch) -> None:
    """Gross errors make the undamped step increase J; the solver rejects it, shrinks
    the radius, switches to the LM bisection and accepts a damped step."""
    spy = _TrustRegionSpy()
    monkeypatch.setattr(wls, "_solve_in_trust_region", spy)
    z = _with_gross_errors(case14_problem["z"], seed=1)
    res = _run(case14_problem, z, tolerance=1e-6, max_iterations=50)

    lm_calls = [c for c in spy.calls if c["lam_floor"] > 0.0]
    assert lm_calls, "no LM attempt: the scenario no longer produces a rejected step"
    # Bisection moved lambda above its floor at least once (the floor step was too long).
    assert any(c["lam_used"] > c["lam_floor"] for c in lm_calls)
    # After a rejection the radius is quartered and lambda starts at lam_init = 1e-3.
    first = spy.calls.index(lm_calls[0])
    assert spy.calls[first]["lam_floor"] == pytest.approx(1e-3)
    assert spy.calls[first]["tr"] == pytest.approx(0.25 * spy.calls[first - 1]["tr"])
    # The run goes on after the recovery (more than one accepted iteration).
    assert res.iterations > 1
    assert np.all(np.isfinite(res.x))
    assert np.isfinite(res.objective)


def test_force_accepted_tiny_step_after_rejections_reports_success(
    case14_problem, monkeypatch
) -> None:
    """Current behaviour, see issue #57: after a chain of rejections the radius drops
    below ``tolerance``; the next (tiny) step is force-accepted without the ratio
    test and the run reports ``success=True`` although that step increased J.

    The rejections are genuine: the iterate reaches a region where a voltage
    magnitude state is negative, where ``h`` (|V|) and its Jacobian disagree, so
    the Gauss-Newton direction stops being a descent direction.
    """
    z = _with_gross_errors(case14_problem["z"], seed=0)
    spy = _TrustRegionSpy()
    monkeypatch.setattr(wls, "_solve_in_trust_region", spy)
    res = _run(case14_problem, z, tolerance=1e-6, max_iterations=50)
    monkeypatch.undo()

    assert res.success  # #57: reported as converged
    assert res.solve_status is SolveStatus.TR_COLLAPSED  # ... but the status tells
    assert res.iterations < 50
    # The last iteration made several attempts with a shrinking radius (rejections).
    last = spy.calls[-1]
    assert last["lam_floor"] > 0.0
    assert last["tr"] < 1e-6
    # J before the last accepted iteration vs after it.
    before = _run(case14_problem, z, tolerance=1e-6, max_iterations=res.iterations - 1)
    assert res.objective > before.objective  # #57: the accepted step increased J


def test_trust_region_collapse_reports_failure(case14_problem, monkeypatch) -> None:
    """With a tolerance below the radius reachable by rejections, the chain of
    rejections ends by lambda exceeding ``lam_max = 1e8``: the solver stops with
    ``success=False`` and returns the last accepted iterate."""
    z = _with_gross_errors(case14_problem["z"], seed=0)
    spy = _TrustRegionSpy()
    monkeypatch.setattr(wls, "_solve_in_trust_region", spy)
    res = _run(case14_problem, z, tolerance=1e-9, max_iterations=50)
    monkeypatch.undo()

    assert not res.success
    assert res.solve_status is SolveStatus.STALLED
    assert res.iterations < 50  # early stop, not the iteration limit
    # lambda grows x10 per rejection from 1e-3; the last attempt ran with 1e8.
    assert spy.calls[-1]["lam_floor"] == pytest.approx(1e8)
    assert spy.calls[-1]["tr"] > 1e-9  # collapse came from lambda, not from the radius
    # The returned state is the last accepted iterate.
    again = _run(case14_problem, z, tolerance=1e-9, max_iterations=res.iterations)
    np.testing.assert_array_equal(res.x, again.x)
    assert res.objective == again.objective


# ---------------------------------------------------------------------------
# Early exits on a 2-bus network
# ---------------------------------------------------------------------------


def _two_bus() -> Working:
    """110 kV line 1-2 (R=6.05 Ohm, X=30.25 Ohm), slack at node 1."""
    m = Working.empty()
    for nid, ntype in ((1, NodeType.SLACK), (2, NodeType.PQ)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
            }
        )
    m.branches.add(
        {
            "id": 10,
            "from_node": 1,
            "to_node": 2,
            "resistance": 6.05,
            "reactance": 30.25,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    return m


def _add(m: Working, mid: int, ot: int, oid: int, mt: int, value: float, side: int = -1) -> None:
    m.measurements.add(
        {
            "id": mid,
            "object_type": ot,
            "object_id": oid,
            "measurement_type": mt,
            "value": value,
            "variance": 0.01,
            "status": True,
            "quality": 0,
            "branch_side": side,
        }
    )


def _setup(m: Working):  # type: ignore[no-untyped-def]
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    ybus, yf, yt = build_ybus(pu)
    z, r_matrix, idx = build_z_and_r(m, m.measurements, pu)
    return pu, layout, ybus, yf, yt, z, r_matrix, idx


def _observed_two_bus(p_mw: float, q_mvar: float) -> Working:
    m = _two_bus()
    _add(m, 1, OBJ_NODE, 1, KIND_VOLTAGE, 110.0)
    _add(m, 2, OBJ_NODE, 2, KIND_VOLTAGE, 108.0)
    _add(m, 3, OBJ_BRANCH, 10, KIND_POWER_P, p_mw, side=0)
    _add(m, 4, OBJ_BRANCH, 10, KIND_POWER_Q, q_mvar, side=0)
    return m


def test_zero_gain_diagonal_stops_before_the_first_step() -> None:
    """A single branch current at flat start has a zero Jacobian row (|I| = 0 is not
    differentiable; the derivative is zeroed), so the gain matrix is all zero and
    the solver stops at once with ``success=False``."""
    m = _two_bus()
    _add(m, 1, OBJ_BRANCH, 10, KIND_CURRENT, 50.0, side=0)
    pu, layout, ybus, yf, yt, z, r_matrix, idx = _setup(m)
    e0 = flat_start(layout)
    res = solve_wls(e0, z, r_matrix, ybus, yf, yt, idx, layout, pu)
    assert not res.success
    assert res.solve_status is SolveStatus.ERROR
    assert res.iterations == 0
    np.testing.assert_array_equal(res.x, e0)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_non_finite_objective_stops_with_failure() -> None:
    """A gross error so large that r^T R^-1 r overflows: the trial objective is not
    finite, the solver stops (``success=False``) and keeps the initial state."""
    m = _observed_two_bus(30.0, 10.0)
    pu, layout, ybus, yf, yt, z, r_matrix, idx = _setup(m)
    z = z.copy()
    z[2] = 1e200  # P flow in p.u.: (1e200)^2 / sigma^2 overflows to inf
    e0 = flat_start(layout)
    res = solve_wls(e0, z, r_matrix, ybus, yf, yt, idx, layout, pu)
    assert not res.success
    assert res.solve_status is SolveStatus.DIVERGED
    assert res.iterations == 0
    np.testing.assert_array_equal(res.x, e0)
    assert not np.isfinite(res.objective)


def test_max_step_below_tolerance_reports_convergence_after_one_step() -> None:
    """Current behaviour (related to issue #57): with ``max_step <= tolerance`` the
    very first step is clamped below the tolerance and force-accepted, so the run
    reports ``success=True`` after one iteration although the state has barely
    moved from the flat start and is far from the minimum."""
    m = _observed_two_bus(30.0, 10.0)
    pu, layout, ybus, yf, yt, z, r_matrix, idx = _setup(m)
    e0 = flat_start(layout)
    reference = solve_wls(e0, z, r_matrix, ybus, yf, yt, idx, layout, pu, tolerance=1e-10)
    assert reference.success
    assert reference.solve_status is SolveStatus.KKT

    res = solve_wls(e0, z, r_matrix, ybus, yf, yt, idx, layout, pu, tolerance=1e-6, max_step=1e-7)
    assert res.success
    assert res.solve_status is SolveStatus.TR_COLLAPSED
    assert res.iterations == 1
    assert np.max(np.abs(res.x - e0)) == pytest.approx(1e-7)
    assert np.max(np.abs(reference.x - e0)) > 1e-3
    assert res.objective > 100.0 * reference.objective


def test_input_shape_checks() -> None:
    m = _observed_two_bus(30.0, 10.0)
    pu, layout, ybus, yf, yt, z, r_matrix, idx = _setup(m)
    e0 = flat_start(layout)
    with pytest.raises(ValueError, match="e_init"):
        solve_wls(np.zeros(layout.size + 1), z, r_matrix, ybus, yf, yt, idx, layout, pu)
    with pytest.raises(ValueError):
        solve_wls(e0, z[:-1], r_matrix, ybus, yf, yt, idx, layout, pu)
    r_short = csr_matrix(r_matrix.toarray()[:-1, :-1])
    with pytest.raises(ValueError):
        solve_wls(e0, z[:-1], r_short, ybus, yf, yt, idx, layout, pu)


def test_two_bus_true_state_is_recovered() -> None:
    """Sanity anchor for the 2-bus fixtures: exact measurements of a known state
    (V2 = 0.98 p.u., delta2 = -0.03 rad) are reproduced by the solver."""
    m = _two_bus()
    pu = model_to_pu(m)
    layout = StateLayout.from_slack(pu.n_bus, pu.slack_idx)
    _ybus, yf, _yt = build_ybus(pu)
    v = np.array([1.0, 0.98])
    delta = np.array([0.0, -0.03])
    volts = v * np.exp(1j * delta)
    s_from = volts[0] * np.conj((yf @ volts)[0]) * 100.0
    m2 = _observed_two_bus(float(s_from.real), float(s_from.imag))
    m2.measurements.update(2, {"value": 0.98 * 110.0})
    pu2, layout2, ybus2, yf2, yt2, z, r_matrix, idx = _setup(m2)
    res = solve_wls(
        flat_start(layout2), z, r_matrix, ybus2, yf2, yt2, idx, layout2, pu2, tolerance=1e-10
    )
    assert res.success
    np.testing.assert_allclose(res.x, pack(delta, v, layout), atol=1e-8)
