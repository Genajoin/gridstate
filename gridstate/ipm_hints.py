"""Convergence hints for slow or non-converged IPM solves.

When an IPM solve takes many iterations, does not converge or falls back from
``ipm_pd`` to the primal solver, :func:`ipm_convergence_hints` turns the
solver's step diagnostics into readable lines for the log:

1. **Step limiters** — the coordinates whose steps most often limited the
   iteration (they set the trust-region clip or the boundary rule, or
   dominated a step cut by the line search), mapped to node id and variable;
2. **weakly determined nodes** — box variables that no measurement sees and
   whose prior is loose (a wide declared range);
3. **largest normalized residuals** — the rows the estimate fits worst, and
   part measurements lying outside the box of their variable.

Each limiter gets a hint on what usually helps. Indices are mapped with the
:class:`~gridstate.preprocessing.ipm_setup.IPMSetup` of the solve; the
solver itself knows nothing about nodes.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gridstate.algorithms.ipm import IPMResult
from gridstate.preprocessing.ipm_setup import IPMSetup
from gridstate.solution_hints import KIND_NAMES, describe_row, residual_hints
from gridstate.units import BASE_MVA, NetworkPU
from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_BOX_PRIOR_QGEN,
    KIND_BOX_PRIOR_QNAG,
)


def _box_sections(setup: IPMSetup) -> list[tuple[int, int, np.ndarray]]:
    lay = setup.layout
    return [
        (KIND_BOX_PRIOR_PGEN, lay.offset_pgen, lay.pgen_node_pos),
        (KIND_BOX_PRIOR_QGEN, lay.offset_qgen, lay.qgen_node_pos),
        (KIND_BOX_PRIOR_PNAG, lay.offset_pnag, lay.pnag_node_pos),
        (KIND_BOX_PRIOR_QNAG, lay.offset_qnag, lay.qnag_node_pos),
    ]


def _variable(setup: IPMSetup, j: int) -> tuple[int, int]:
    """``(node position, kind)`` of state index ``j``; kind ``-1`` = δ, ``-2`` = V."""
    lay = setup.layout
    if lay.offset_delta <= j < lay.offset_delta + lay.n_bus - 1:
        return int(lay.non_slack_idx[j - lay.offset_delta]), -1
    if lay.offset_v <= j < lay.offset_v + lay.n_bus:
        return j - lay.offset_v, -2
    for kind, off, pos in _box_sections(setup):
        if off <= j < off + pos.size:
            return int(pos[j - off]), kind
    raise IndexError(f"state index {j} is outside the layout")


class _Context:
    """Lookups shared by the hint builders."""

    def __init__(self, setup: IPMSetup, network_pu: NetworkPU) -> None:
        self.setup = setup
        self.net = network_pu
        mi = setup.meas_index
        self.sigma = np.sqrt(setup.r_matrix.diagonal())
        self.is_role = np.zeros(len(mi), dtype=bool)
        self.is_role[setup.role_rows] = True
        self.box_of = {int(j): k for k, j in enumerate(setup.box_idx_in_state)}
        nbr: list[set[int]] = [set() for _ in range(network_pu.n_bus)]
        for f, t in zip(network_pu.from_idx.tolist(), network_pu.to_idx.tolist(), strict=True):
            if f != t:
                nbr[f].add(t)
                nbr[t].add(f)
        self.neighbours = nbr

    def node_id(self, pos: int) -> int:
        return int(self.net.bus_ids[pos])

    def rows(self, pos: int, kind: int) -> np.ndarray:
        mi = self.setup.meas_index
        return np.flatnonzero((mi.object_kind == 0) & (mi.object_pos == pos) & (mi.kind == kind))

    def box_variable_state(self, pos: int, kind: int) -> tuple[bool, float]:
        """``(measured, prior σ in MW / MVAr)`` of a box variable (σ = nan without a prior)."""
        rows = self.rows(pos, kind)
        measured = bool(self.is_role[rows].any())
        prior = rows[~self.is_role[rows]]
        sigma = float(self.sigma[prior[0]] * BASE_MVA) if prior.size else float("nan")
        return measured, sigma

    def describe(self, j: int) -> tuple[str, str]:
        """``(label, hint)`` of state index ``j``."""
        pos, kind = _variable(self.setup, j)
        node = self.node_id(pos)
        if kind < 0:
            var = "δ" if kind == -1 else "V"
            n_nbr = len(self.neighbours[pos])
            free_parts = []
            for bkind, _off, _pos in _box_sections(self.setup):
                rows = self.rows(pos, bkind)
                if rows.size and not self.is_role[rows].any():
                    free_parts.append(KIND_NAMES[bkind])
            label = f"node {node} {var} ({n_nbr} neighbour node{'s' if n_nbr != 1 else ''})"
            if n_nbr <= 1 and free_parts:
                hint = (
                    f"radial node with unmeasured {'/'.join(free_parts)}: its state and "
                    "injection are fixed by the balance alone; measure the injection or "
                    "the feeding branch flow, or narrow the declared ranges"
                )
            else:
                hint = "weakly observed state: check measurements and topology around the node"
            return label, hint
        k = self.box_of.get(j)
        lo = float(self.setup.box_lo[k]) * BASE_MVA if k is not None else float("nan")
        hi = float(self.setup.box_hi[k]) * BASE_MVA if k is not None else float("nan")
        measured, sigma = self.box_variable_state(pos, kind)
        name = KIND_NAMES[kind]
        label = f"node {node} {name} range [{lo:.1f}, {hi:.1f}]"
        if measured:
            label += ", measured"
            hint = "the estimate presses against the range: check the range against the measurement"
        else:
            label += f", unmeasured, prior σ {sigma:.1f}" if np.isfinite(sigma) else ", unmeasured"
            hint = (
                "nearly free variable: narrow the declared range or add a measurement"
                if np.isfinite(sigma) and sigma >= 100.0
                else "the estimate presses against the range: check the declared range"
            )
        return label, hint

    def describe_row(self, row: int) -> str:
        mi = self.setup.meas_index
        return describe_row(mi, self.net, row, part=bool(self.is_role[row]))


def _top(counts: Mapping[int, int], n: int, floor: float) -> list[tuple[int, int]]:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [kv for kv in ranked if kv[1] >= floor][:n]


def has_dominant_limiter(result: IPMResult, iterations: int, min_share: float = 0.1) -> bool:
    """Whether some coordinate or measurement bound limited at least ``min_share`` of the steps.

    A solve can be long without one: the primal IPM walks its barrier schedule
    in many short inner solves by design.
    """
    floor = min_share * max(iterations, 1)
    counts = list(result.step_limiters.values()) + list(result.limiting_rows.values())
    return bool(counts) and max(counts) >= floor


def ipm_convergence_hints(
    *,
    setup: IPMSetup,
    network_pu: NetworkPU,
    result: IPMResult,
    residual: np.ndarray,
    iterations: int,
    top: int = 5,
    weak_sigma_mw: float = 100.0,
    min_share: float = 0.1,
) -> list[str]:
    """Hint lines for a slow / non-converged solve (see the module docstring).

    Args:
        setup: the IPM setup of the solve (index → node / measurement mapping).
        network_pu: the network of the solve.
        result: the slow solve; its ``step_limiters`` / ``limiting_rows`` are used.
        residual: ``r = z − h(x)`` at the final estimate (rows of ``setup.z``).
        iterations: the iteration count reported for the slow solve.
        top: entries per section.
        weak_sigma_mw: prior σ (MW / MVAr) from which an unmeasured box variable
            counts as weakly determined.
        min_share: a limiter is listed when it limited at least this share of
            the iterations.
    """
    ctx = _Context(setup, network_pu)
    lines: list[str] = []
    floor = min_share * max(iterations, 1)

    limiters = _top(result.step_limiters, top, floor)
    if limiters:
        lines.append("step limited most often by:")
        for j, cnt in limiters:
            label, hint = ctx.describe(j)
            lines.append(f"  {label}: {cnt} of {iterations} iterations — {hint}")
    rows = _top(result.limiting_rows, top, floor)
    if rows:
        lines.append("measurement bounds that limited the step:")
        for row, cnt in rows:
            lines.append(
                f"  {ctx.describe_row(row)}: {cnt} of {iterations} iterations — "
                "check its min/max against the estimate"
            )

    weak: list[tuple[float, str]] = []
    for kind, off, pos in _box_sections(setup):
        for jj, p in enumerate(pos.tolist()):
            measured, sigma = ctx.box_variable_state(p, kind)
            if not measured and np.isfinite(sigma) and sigma >= weak_sigma_mw:
                label, _ = ctx.describe(off + jj)
                weak.append((sigma, label))
    if weak:
        weak.sort(key=lambda t: -t[0])
        lines.append(
            f"{len(weak)} unmeasured box variable(s) with prior σ ≥ {weak_sigma_mw:g} "
            "(nearly free; a wide declared range):"
        )
        lines.extend(f"  {label}" for _, label in weak[:top])

    sigma = ctx.sigma
    lines.extend(
        residual_hints(
            residual,
            sigma,
            meas_index=setup.meas_index,
            network_pu=network_pu,
            # Equality rows: their σ is a penalty, not a measurement error.
            skip=setup.balance_rows if setup.balance_tol is not None else None,
            part_rows=setup.role_rows,
            top=top,
        )
    )

    outside: list[tuple[float, str]] = []
    for row in setup.role_rows.tolist():
        kind, p = int(setup.meas_index.kind[row]), int(setup.meas_index.object_pos[row])
        for bkind, off, pos in _box_sections(setup):
            if bkind != kind:
                continue
            hit = np.flatnonzero(pos == p)
            k = ctx.box_of.get(off + int(hit[0])) if hit.size else None
            if k is None:
                continue
            lo, hi, zz = setup.box_lo[k], setup.box_hi[k], setup.z[row]
            d = max(lo - zz, zz - hi, 0.0) / sigma[row] if sigma[row] > 0 else 0.0
            if d > 3.0:
                outside.append(
                    (
                        d,
                        f"  {ctx.describe_row(row)}: z = {zz * BASE_MVA:.1f} outside "
                        f"[{lo * BASE_MVA:.1f}, {hi * BASE_MVA:.1f}] by {d:.1f}σ",
                    )
                )
    if outside:
        outside.sort(key=lambda t: -t[0])
        lines.append("part measurements outside the range of their variable:")
        lines.extend(text for _, text in outside[:top])
    return lines


__all__ = ["has_dominant_limiter", "ipm_convergence_hints"]
