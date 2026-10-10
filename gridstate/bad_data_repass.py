"""Two-pass bad-data mechanism on the residuals of a solved SE.

After the first solve the residuals ``z − h`` (``h`` = ``estimated_si``) give a
reliable diagnosis of gross telemetry errors: sign flips, "broken zeros"
(z=0 at large h), isolated monster injections. The classification below
builds a correction plan (flip / reject / damp), and the pipeline applies it to the
working model and re-solves (warm, ``init="results"``).

The configuration is validated on four large production models with one set of
options (no per-model tuning): T=10 on σ_det=min(σ,30), pair immunity of
consistent branch pairs (an anti-circular guard: in a bad solution the largest
residuals may belong to HONEST measurements, anchors that must not be rejected),
the relative flip criterion |z+h| < γ·|z−h|, the (node, P/Q domain) coverage
guard and damping of Q injections instead of rejecting them (a blind boundary
node without Qinj settles on a garbage pseudo-prior through the Q balance).

The detection σ_det catches conflicts hidden by the giant σ≈α·|z| of large
flows; the solver WEIGHTS are not changed (capping weights ruins accuracy).
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import meas_mask
from gridstate.utils import branch_endpoints_map
from gridstate.working import table_copy, table_replace
from gridstate.z_vector import (
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
)


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)

# Kinds of real measurements taking part in detection (V does not: V measurements carry
# no sign flips, and rejecting them destroys voltage observability).
DETECTABLE_KINDS = (
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
)

# Pair immunity: consistency tolerance |z_b + z_e| relative to max|z|
# (P flows are more precise than Q; Q pairs diverge by branch charging/losses).
PAIR_TOL_P = 0.08
PAIR_TOL_Q = 0.30
# Below this |z| (MW/MVAr) a pair is not evidence (zero pairs
# get no immunity: z=0 proves nothing).
PAIR_Z_MIN = 5.0
# Pair consistency floor: 5·max(σ_det) of the two sides.
PAIR_SIGMA_MULT = 5.0

# Domain coverage of a node: V is deliberately NOT counted, since a V measurement does not save
# P/Q observability (after an injection is removed the node settles on pseudo-priors,
# which are garbage at the network edge).
_FLOW_KINDS = (KIND_POWER_P, KIND_POWER_Q, KIND_CURRENT)
_DOMAIN = {
    KIND_POWER_P: "P",
    KIND_CURRENT: "P",
    KIND_POWER_INJECTION_P: "P",
    KIND_POWER_Q: "Q",
    KIND_POWER_INJECTION_Q: "Q",
}


@dataclass(frozen=True)
class BadDataPlan:
    """Plan of measurement corrections, based on the residual classification."""

    flip_ids: frozenset[int]  # value := −value (channel sign flip)
    reject_ids: frozenset[int]  # status := False
    damp_ids: frozenset[int]  # variance := variance · k² (Qinj damping)
    n_candidates: int  # total candidates (rn_det > T)
    n_immune: int  # of which saved by pair immunity
    n_restored: int  # restored by the coverage guard

    @property
    def empty(self) -> bool:
        return not (self.flip_ids or self.reject_ids or self.damp_ids)


def _real_detectable(m: np.ndarray) -> np.ndarray:
    """Mask of real measurements usable for detection (active, not pseudo, h present)."""
    return meas_mask(m, kinds=DETECTABLE_KINDS, pseudo=False, finite_estimate=True)


def classify_bad_data(
    measurements: np.ndarray,
    branches: np.ndarray,
    *,
    threshold: float,
    sigma_cap: float,
    flip_ratio: float,
) -> BadDataPlan:
    """Classify real measurements by the residuals of a solved SE.

    Args:
        measurements: structured measurements array of the working model AFTER
            the solve (``estimated_si`` filled by write_measurement_estimates).
        branches: structured branches array (needed by the coverage guard).
        threshold: T, the candidate threshold ``|z−h|/σ_det > T``.
        sigma_cap: cap of the detection σ (MW/MVAr); <=0 or inf means no cap.
        flip_ratio: γ, flip if ``|z+h| < γ·|z−h|`` (and |z|>0).

    Returns:
        :class:`BadDataPlan` (may be ``empty``; then no re-solve is needed).
    """
    sel = _real_detectable(measurements)
    z = measurements["value"][sel]
    h = measurements["estimated_si"][sel]
    sig = np.sqrt(measurements["variance"][sel])
    ids = measurements["id"][sel]
    kinds = measurements["measurement_type"][sel]
    oid = measurements["object_id"][sel]
    side = measurements["branch_side"][sel]

    capped = sigma_cap > 0 and np.isfinite(sigma_cap)
    sig_det = np.minimum(sig, sigma_cap) if capped else sig
    rn_det = np.abs(z - h) / np.maximum(sig_det, 1e-9)
    cand = rn_det > threshold

    # Pair immunity: a from/to branch pair with consistent z is an honest
    # flow (possibly a victim of a bad solution); leave it alone.
    immune = np.zeros(len(z), dtype=bool)
    for k, tol in ((KIND_POWER_P, PAIR_TOL_P), (KIND_POWER_Q, PAIR_TOL_Q)):
        km = np.where(kinds == k)[0]
        by_obj: dict[int, dict[int, list[int]]] = collections.defaultdict(lambda: {0: [], 1: []})
        for j in km:
            s = int(side[j])
            if s in (0, 1):
                by_obj[int(oid[j])][s].append(int(j))
        for ss in by_obj.values():
            if not ss[0] or not ss[1]:
                continue
            zb = float(np.median(z[ss[0]]))
            ze = float(np.median(z[ss[1]]))
            zmax = max(abs(zb), abs(ze))
            if zmax < PAIR_Z_MIN:
                continue
            smax = float(max(sig_det[ss[0] + ss[1]].max(), 1e-9))
            if abs(zb + ze) <= max(tol * zmax, PAIR_SIGMA_MULT * smax):
                for j in ss[0] + ss[1]:
                    immune[j] = True

    flip_mask = cand & ~immune & (np.abs(z + h) < flip_ratio * np.abs(z - h)) & (np.abs(z) > 1e-9)
    rej_mask = cand & ~immune & ~flip_mask

    flip_ids = {int(i) for i in ids[flip_mask]}
    rej_ids = {int(i) for i in ids[rej_mask]}

    # --- Coverage guard (node, domain): reject is allowed only if every
    # affected node keeps at least one real measurement of its P/Q domain
    # (an injection at the node or a flow on the node's side).
    b2n = branch_endpoints_map(branches)
    cover: dict[tuple[int, str], set[int]] = collections.defaultdict(set)
    for j in np.where(sel)[0]:
        mid = int(measurements["id"][j])
        k = int(measurements["measurement_type"][j])
        o = int(measurements["object_id"][j])
        dom = _DOMAIN.get(k)
        if dom is None:
            continue
        if k in _FLOW_KINDS:
            ends = b2n.get(o)
            if ends is None:
                continue
            s = int(measurements["branch_side"][j])
            nodes = (ends[s],) if s in (0, 1) else ends
        else:
            nodes = (o,)
        for n in nodes:
            cover[(n, dom)].add(mid)
    restored: set[int] = set()
    for ms in cover.values():
        if ms and ms.issubset(rej_ids):
            restored |= ms

    # --- Qinj is never rejected, only damped by σ·k (a node without Qinj settles
    # on a pseudo-prior through the Q balance; formal flow coverage does not save it).
    qinj_sel = sel & (measurements["measurement_type"] == KIND_POWER_INJECTION_Q)
    qinj_ids = {int(i) for i in measurements["id"][qinj_sel]}
    damp_ids = rej_ids & qinj_ids
    rej_ids = (rej_ids - damp_ids) - restored

    return BadDataPlan(
        flip_ids=frozenset(flip_ids),
        reject_ids=frozenset(rej_ids),
        damp_ids=frozenset(damp_ids),
        n_candidates=int(cand.sum()),
        n_immune=int((cand & immune).sum()),
        n_restored=len(restored - damp_ids),
    )


def apply_bad_data_plan(model: Working, plan: BadDataPlan, *, damp_factor: float) -> dict:
    """Apply the plan to ``model.measurements`` (the pipeline's working copy).

    flip → ``value := −value``; reject → ``status := False``;
    damp → ``variance := variance · damp_factor²`` (σ × damp_factor).
    The solver builds weights from variance (see ``z_vector.build_z_and_r``).
    """
    m = table_copy(model.measurements)
    if plan.flip_ids:
        sel = np.isin(m["id"], list(plan.flip_ids))
        m["value"][sel] *= -1.0
    if plan.reject_ids:
        sel = np.isin(m["id"], list(plan.reject_ids))
        m["status"][sel] = False
    if plan.damp_ids:
        sel = np.isin(m["id"], list(plan.damp_ids))
        m["variance"][sel] *= float(damp_factor) ** 2
    table_replace(model.measurements, m)
    return {
        "flips": len(plan.flip_ids),
        "rejects": len(plan.reject_ids),
        "damped": len(plan.damp_ids),
        "candidates": plan.n_candidates,
        "immune": plan.n_immune,
        "restored": plan.n_restored,
    }
