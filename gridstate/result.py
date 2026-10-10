"""State estimation result — returned by ``estimate()``.

A reference to the working model (the ``working`` layer) is stored in ``SEResult.model``.
When called via :func:`gridstate.pipeline.run` this is a **copy** of the input model:
the input stays read-only, the output lives in the working copy and in ``SEResult``. On a
direct ``estimate(model)`` call the model is updated in place (the low-level
engine), and ``SEResult.model`` is the same reference.

**Output layer (``outputs``)** — :class:`OutputTables` with structured
arrays (nodes/branches/measurements), keyed by id, parallel to the Input collections. This is the
canonical Output contract for external adapters (UI/CLI): they join Input +
``outputs`` by id for display. The fields ``v_pu`` / ``delta_rad`` duplicate V/δ in
p.u./radians — for debugging and tests.

The additional quality fields (``chi2``, ``worst_residuals``,
``worst_imbalance``, ``observability_warnings``) are filled by ``estimate()``
after ``write_results_to_model``: a structured summary for backward-compatible
diagnostics (it used to be assembled ad hoc by an example script).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gridstate.algorithms.status import SolveStatus
from gridstate.working import table_view


if TYPE_CHECKING:
    from gridstate.working import Working


# Solver output fields per model layer — the ``OutputTables`` Output container
# extracts exactly these from the working model after the results are written.
NODE_OUTPUT_FIELDS: tuple[str, ...] = (
    "voltage_magnitude",
    "voltage_angle",
    "p_inj_calc",
    "q_inj_calc",
    "imbalance_p",
    "imbalance_q",
    "load_p_estimated",
    "load_q_estimated",
    "generation_p_estimated",
    "generation_q_estimated",
    "solved",
)
BRANCH_OUTPUT_FIELDS: tuple[str, ...] = (
    "power_from_p",
    "power_from_q",
    "power_to_p",
    "power_to_q",
    "current_from",
    "current_to",
    "loss_p",
    "loss_q",
    "loading_pct",
)
MEAS_OUTPUT_FIELDS: tuple[str, ...] = (
    "estimated_si",
    "residual",
)


@dataclass
class Chi2Summary:
    """Summary of the χ² test on the final SE solution.

    Attributes:
        value: ``J = Σ(r_i² / σ_i²)`` — the WLS objective at convergence.
        dof: degrees of freedom ``m − n`` (number of measurements minus the state size).
        threshold: critical value ``χ²(dof, 1 − α)``, ``α = 0.05``.
            ``NaN`` for ``dof ≤ 0`` (the test is inconclusive).
        passes: ``True`` if ``value ≤ threshold`` (no signs of bad data).
    """

    value: float
    dof: int
    threshold: float
    passes: bool


@dataclass
class ResidualRow:
    """A single entry of the top worst normalized residuals.

    Attributes:
        measurement_id: ``Measurement.id`` of the source measurement.
        kind: text label of the type (``"V"``/``"P"``/``"Q"``/``"I"``/
            ``"P_inj"``/``"Q_inj"``/``"?"``).
        value: measurement value ``z`` in p.u. (original units are in
            ``measurements.value``/``estimated_si`` by ``measurement_id``).
        expected: model value ``h(x)`` in p.u.
        residual: ``value − expected`` in p.u.
        normalized_residual: ``|r| / √diag(Ω)`` — normalized residual
            (Abur & Expósito §5.6). ``inf`` for non-redundant measurements.
        sigma: measurement σ in p.u. (``√σ²`` from the R matrix); ``nan`` if
            unavailable.
        object_kind: 0=node / 1=branch / 2=generator; ``-1`` — unknown.
        object_id: ``id`` of the measured object (node/branch); ``0`` if
            unknown.
        branch_side: branch side (0=from, 1=to, ``-1`` — not a branch).
        is_pseudo: pseudo-prior (``add_pseudo``/synthesis), not a telemetry measurement.
    """

    measurement_id: int
    kind: str
    value: float
    expected: float
    residual: float
    normalized_residual: float
    sigma: float = float("nan")
    object_kind: int = -1
    object_id: int = 0
    branch_side: int = -1
    is_pseudo: bool = False


@dataclass
class NearCriticalRow:
    """A measurement whose ``Ω_ii`` is a small fraction of its variance.

    ``Ω_ii = σ²_i − (H G⁻¹ Hᵀ)_ii`` is then a difference of two nearly equal
    numbers: the rounding of ``(H G⁻¹ Hᵀ)_ii`` is amplified by
    ``1 / omega_ratio`` in ``Ω_ii`` and in the normalized residual, and two
    exact methods of computing it may disagree in the leading digits. The
    measurement is nearly critical: the estimate follows it almost exactly,
    so its residual says little about its error. Such measurements are left
    out of ``SEResult.worst_residuals`` and listed here instead.

    Attributes:
        measurement_id: ``Measurement.id`` of the source measurement.
        kind: text label of the type (as in :class:`ResidualRow`).
        omega_ratio: ``Ω_ii / σ²_i`` (below the configured threshold).
        normalized_residual: ``|r| / √Ω_ii`` as computed (``inf`` if
            ``Ω_ii`` is numerically non-positive).
        is_pseudo: pseudo-prior, not a telemetry measurement. Always
            ``False`` under the default residual scope ``"real"``, which
            leaves pseudo-measurements out of ``r_N`` (and so out of this
            report); it can be ``True`` only with
            ``quality_summary_scope="all"``.
        object_kind: 0=node / 1=branch / 2=generator; ``-1`` — unknown
            (as in :class:`ResidualRow`).
        object_id: ``id`` of the measured object (node/branch); ``0`` if
            unknown.
        branch_side: branch side (0=from, 1=to, ``-1`` — not a branch).
    """

    measurement_id: int
    kind: str
    omega_ratio: float
    normalized_residual: float
    is_pseudo: bool = False
    object_kind: int = -1
    object_id: int = 0
    branch_side: int = -1


@dataclass
class ImbalanceRow:
    """A single entry of the top nodal imbalances.

    Attributes:
        node_id: ``Node.id`` of the node.
        imbalance_p_mw: ``p_inj_calc − (generation_p − load_p)`` in MW.
        imbalance_q_mvar: likewise for reactive power.
    """

    node_id: int
    imbalance_p_mw: float
    imbalance_q_mvar: float


# Text labels for ``ResidualRow.kind`` — kept in sync with
# ``gridstate.z_vector.KIND_*`` (repeated here as literals so as not to
# pull an import into result.py).
_RESIDUAL_KIND_LABELS: dict[int, str] = {
    0: "P",  # KIND_POWER_P
    1: "Q",  # KIND_POWER_Q
    2: "V",  # KIND_VOLTAGE
    3: "I",  # KIND_CURRENT
    4: "P_inj",  # KIND_POWER_INJECTION_P
    5: "Q_inj",  # KIND_POWER_INJECTION_Q
    6: "Pbal",  # KIND_NODE_BALANCE_P (IPM)
    7: "Qbal",  # KIND_NODE_BALANCE_Q (IPM)
    8: "Pg_pr",  # KIND_BOX_PRIOR_PGEN (IPM)
    9: "Qg_pr",  # KIND_BOX_PRIOR_QGEN (IPM)
    10: "Pn_pr",  # KIND_BOX_PRIOR_PNAG (IPM)
    11: "Qn_pr",  # KIND_BOX_PRIOR_QNAG (IPM)
}


def _select_output_fields(arr: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    """Build a structured array ``id`` + ``fields`` from a collection.

    Only the fields actually present in ``arr.dtype`` are taken (robust to
    different DTYPE versions). If the array is empty or has no ``id`` — an empty result.
    """
    names = set(arr.dtype.names or ())
    if "id" not in names:
        return np.empty(0, dtype=[("id", "i8")])
    present = [f for f in fields if f in names]
    out_dtype = [("id", "i8")] + [(f, "f8") for f in present]
    out = np.empty(len(arr), dtype=out_dtype)
    out["id"] = arr["id"].astype(np.int64, copy=False)
    for f in present:
        out[f] = arr[f].astype(np.float64, copy=False)
    return out


@dataclass
class OutputTables:
    """SE Output layer: results keyed by id, parallel to the Input collections.

    The canonical Output contract (Input is read-only): external adapters (UI/CLI)
    join Input + these tables by ``id`` for display (nodes/branches on a single
    page — input fields on the left, result on the right). Extracted from the working
    model after the solver results are written.

    Attributes:
        nodes: structured array ``id`` + :data:`NODE_OUTPUT_FIELDS`
            (voltage_magnitude/voltage_angle/p_inj_calc/q_inj_calc/imbalance_*/
            load_*_estimated/generation_*_estimated).
        branches: ``id`` + :data:`BRANCH_OUTPUT_FIELDS`
            (power_from/to_p/q, current_from/to, loss_p/q, loading_pct).
        measurements: ``id`` + :data:`MEAS_OUTPUT_FIELDS`
            (estimated_si, residual).
    """

    nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[("id", "i8")]))
    branches: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[("id", "i8")]))
    measurements: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=[("id", "i8")]))

    def node(self, node_id: int) -> dict | None:
        """Output row of a node as a dict (or ``None`` if absent)."""
        return self._row(self.nodes, node_id)

    def branch(self, branch_id: int) -> dict | None:
        """Output row of a branch as a dict (or ``None``)."""
        return self._row(self.branches, branch_id)

    def measurement(self, meas_id: int) -> dict | None:
        """Output row of a measurement as a dict (or ``None``)."""
        return self._row(self.measurements, meas_id)

    @staticmethod
    def _row(arr: np.ndarray, obj_id: int) -> dict | None:
        if arr.size == 0:
            return None
        hits = np.where(arr["id"] == int(obj_id))[0]
        if hits.size == 0:
            return None
        row = arr[int(hits[0])]
        names = arr.dtype.names or ()
        return {name: row[name].item() for name in names}


def extract_output_tables(model: Working) -> OutputTables:
    """Extract :class:`OutputTables` from the working model after the results are written."""
    return OutputTables(
        nodes=_select_output_fields(table_view(model.nodes), NODE_OUTPUT_FIELDS),
        branches=_select_output_fields(table_view(model.branches), BRANCH_OUTPUT_FIELDS),
        measurements=_select_output_fields(table_view(model.measurements), MEAS_OUTPUT_FIELDS),
    )


@dataclass
class SEResult:
    """Result of a single ``estimate()`` run.

    Attributes:
        model: reference to the updated ``Working`` (the same object that
            was passed to ``estimate``).
        success: whether the iterative process converged.
        iterations: number of iterations performed.
        objective_value: the solver-internal objective at the last iteration
            (WLS: ``rᵀ R⁻¹ r`` with the Huber weights; IPM: ``½ rᵀ R⁻¹ r`` over
            the augmented vector incl. balance/prior rows). Not comparable
            between the solvers — use ``objective_wls``.
        solve_status: solver-independent termination status
            (:class:`~gridstate.algorithms.status.SolveStatus`).
        objective_wls: plain WLS objective ``Σ (z − h)² / σ²`` of the
            measurement vector at the final state (comparable between WLS
            and IPM).
        algorithm: name of the algorithm used ("wls" / "ipm").
        v_pu: (n_nodes,) — voltage magnitudes in p.u. (duplicating
            ``model.nodes.voltage_magnitude / voltage_nominal``).
        delta_rad: (n_nodes,) — voltage angles in radians.
        message: free-form diagnostic message (reason for non-convergence
            etc.).
        chi2: summary of the χ² test (``None`` if not computed).
        worst_residuals: top-N measurements by ``|r_N|`` (descending).
            Critical measurements (``r_N = inf``) and the nearly critical
            ones of ``near_critical_measurements`` are left out: their
            ``r_N`` is undefined or numerically unreliable. Empty list if
            the computation was skipped / there are no measurements.
        worst_imbalance: top-N nodes by ``|imbalance_p|`` (descending).
        observability_warnings: list of ``node_id`` of nodes with zero
            ``H`` columns (state variables not covered by measurements).
            Empty if the analysis was not run.
        near_critical_measurements: measurements (in the scope of
            ``worst_residuals``) whose ``Ω_ii / σ²_i`` is below the
            near-critical threshold, ascending by that ratio (least
            reliable first; critical ones, ``r_N = inf``, are included);
            their ``normalized_residual`` is numerically unreliable (see
            :class:`NearCriticalRow`), so they are listed here instead of
            in ``worst_residuals``. A threshold of ``0`` lists none and
            ranks every finite ``r_N``. Reporting only: the ``r_N`` values
            and the bad-data test are not affected.
            Pseudo-measurements are excluded from ``r_N`` by construction
            under the default scope ``"real"`` (their ratio is not
            computed), so they never appear here and ``is_pseudo`` is always
            ``False``; with ``quality_summary_scope="all"`` they take part
            and are marked by ``is_pseudo`` (the pipeline always uses
            ``"real"``).
    """

    model: Working
    success: bool = False
    iterations: int = 0
    objective_value: float = float("nan")
    algorithm: str = ""
    v_pu: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta_rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    message: str = ""
    # Convergence breakdown. WLS: "converged"/"not_converged". IPM is
    # two-level (see IPMResult.status): "kkt" (strict stationarity),
    # "completed" (the μ schedule was run through, the solution is usable), "stalled",
    # "error". success == True ⇔ status ∈ {kkt, completed, converged} under the
    # default success_policy="legacy".
    convergence_status: str = ""
    # Solver-independent termination status (see gridstate.algorithms.status;
    # None when the result did not come from a solver run).
    solve_status: SolveStatus | None = None
    # Plain WLS objective Σ (z − h)² / σ² of the measurement vector at the final
    # V/δ: no Huber weights, no ½, no IPM balance/prior rows. Unlike
    # objective_value (the solver-internal objective) it is comparable between
    # WLS and IPM.
    objective_wls: float = float("nan")

    # Output container — results keyed by id (nodes/branches/measurements), parallel to
    # the Input collections. The canonical Output contract for adapters (Input is
    # read-only). Filled by ``estimate()`` after the results are written.
    outputs: OutputTables = field(default_factory=OutputTables)

    # Quality summary — new optional fields; the default is empty/None
    # for backward compatibility.
    chi2: Chi2Summary | None = None
    worst_residuals: list[ResidualRow] = field(default_factory=list)
    worst_imbalance: list[ImbalanceRow] = field(default_factory=list)
    observability_warnings: list[int] = field(default_factory=list)
    near_critical_measurements: list[NearCriticalRow] = field(default_factory=list)


def label_for_kind(kind_code: int) -> str:
    """Get a text label for the numeric ``MeasurementIndex.kind``."""
    return _RESIDUAL_KIND_LABELS.get(int(kind_code), "?")
