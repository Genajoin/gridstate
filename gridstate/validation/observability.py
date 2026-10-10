"""Network observability analysis for a given set of measurements.

The approach is numerical, via the rank of the Jacobian ``H`` at a flat start:

1. ``H = ∂h/∂E`` is built for all active measurements at ``V=1`` p.u., ``δ=0``.
2. The network is observable iff ``rank(H) == 2·n_bus − 1``.
3. If the rank is deficient, for diagnostics we compute the column norms of ``H``
   and flag the nodes whose state variables ``δ_i / V_i`` are not covered
   by any measurement (a zero column).

This is not the full analysis scheme from Abur & Expósito (§4–5), but it covers
most practical cases of unobservability (missing measurements
at individual nodes/islands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gridstate.algebra.base import BaseAlgebra
from gridstate.state import StateLayout, flat_start, unpack
from gridstate.units import model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r


if TYPE_CHECKING:
    from gridstate.working import Working, _ArrayCollection


@dataclass
class ObservabilityReport:
    """Result of the observability analysis.

    Attributes:
        is_observable: ``True`` if ``rank(H) == 2·n_bus − 1``.
        n_state_vars: size of the state vector = ``2·n_bus − 1``.
        n_measurements: number of active measurements (after filtering
            ``status``/``quality``).
        rank_H: actual rank of H at the flat start.
        unobservable_buses: IDs of nodes whose δ or V is not covered by any
            measurement (zero column of H).
        unobservable_branches: IDs of branches for which no measurement
            could be assembled (reserved — always empty for now).
        diagnostics: human-readable message for logs.
    """

    is_observable: bool
    n_state_vars: int
    n_measurements: int
    rank_H: int
    unobservable_buses: list[int] = field(default_factory=list)
    unobservable_branches: list[int] = field(default_factory=list)
    diagnostics: str = ""


def analyze_observability(
    model: Working,
    measurements: _ArrayCollection | None = None,
) -> ObservabilityReport:
    """Check observability and return a report.

    Args:
        model: ``Working``.
        measurements: measurement collection. ``None`` → ``model.measurements``.

    Returns:
        ``ObservabilityReport``. If ``n_measurements < n_state_vars`` the
        necessary condition is violated, but the rank analysis is still run
        anyway (rank will be ≤ n_meas).
    """
    if measurements is None:
        measurements = model.measurements

    network_pu = model_to_pu(model)
    ybus, yf, yt = build_ybus(network_pu)
    z, _, meas_index = build_z_and_r(model, measurements, network_pu)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)

    n_state = layout.size
    n_meas = int(z.shape[0])

    if n_meas == 0:
        return ObservabilityReport(
            is_observable=False,
            n_state_vars=n_state,
            n_measurements=0,
            rank_H=0,
            unobservable_buses=network_pu.bus_ids.tolist(),
            diagnostics="No active measurements.",
        )

    algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
    delta, v = unpack(flat_start(layout), layout)
    H = algebra.evaluate_jacobian(v, delta)
    H_dense = H.toarray()

    rank_h = int(np.linalg.matrix_rank(H_dense))
    is_obs = rank_h == n_state

    # Diagnostics: which columns of H are empty?
    col_norms = np.linalg.norm(H_dense, axis=0)
    zero_cols = np.where(col_norms < 1e-10)[0]

    unobs_buses: set[int] = set()
    n_bus = network_pu.n_bus
    for j in zero_cols:
        # j < n_bus-1 → column δ_non_slack[j]; otherwise → column V[j − (n_bus − 1)]
        bus_pos = int(layout.non_slack_idx[j]) if j < n_bus - 1 else int(j - (n_bus - 1))
        unobs_buses.add(int(network_pu.bus_ids[bus_pos]))

    if is_obs:
        diagnostics = (
            f"Network is observable: rank(H)={rank_h} = n_state={n_state}, m={n_meas} measurements."
        )
    else:
        diagnostics = (
            f"Network is NOT observable: rank(H)={rank_h} < n_state={n_state} "
            f"(m={n_meas}). Rank deficiency = {n_state - rank_h}."
        )
        if unobs_buses:
            diagnostics += f" Nodes without coverage in H: {sorted(unobs_buses)}."

    return ObservabilityReport(
        is_observable=is_obs,
        n_state_vars=n_state,
        n_measurements=n_meas,
        rank_H=rank_h,
        unobservable_buses=sorted(unobs_buses),
        diagnostics=diagnostics,
    )
