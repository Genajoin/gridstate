"""Runtime containers of the contract: :class:`SEInput` and :class:`SEOutput`.

:class:`SEInput` carries the working layer plus the precomputed numeric plans
(:class:`~gridstate.contract.derived.DerivedInputs`); :class:`SEOutput` wraps
the ``SEResult`` of a run into the output tables keyed by id. The facade that
runs the pipeline on them (:func:`gridstate.runtime.run`) lives outside the
contract package, which stays a leaf: importing it never imports the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.contract.tables import SE_OUTPUT
from gridstate.contract.validate import ValidationReport, validate_input
from gridstate.contract.version import CONTRACT_VERSION


if TYPE_CHECKING:
    from gridstate.contract.derived import DerivedInputs
    from gridstate.result import SEResult


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass
class SEInput:
    """SE input contract (runtime container).

    Attributes:
        model: working layer / carrier of the input tables (read-only source).
        derived: precomputed numeric plans (:class:`~gridstate.contract.derived.
            DerivedInputs`) — topology/tap changers/telemetry/materialization/Vnom, ready
            to be applied by the contract kernels. ``None`` — the corresponding steps
            are skipped (the model must already carry the measurements).
        contract_version: contract version the data was built under.
    """

    model: Any
    derived: DerivedInputs | None = None
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        derived: DerivedInputs | None = None,
        contract_version: str = CONTRACT_VERSION,
    ) -> SEInput:
        """Wrap an existing model into an input contract."""
        return cls(
            model=model,
            derived=derived,
            contract_version=contract_version,
        )

    def validate(self, *, strict: bool = False) -> ValidationReport:
        """Validate the input data against the contract (see :func:`validate_input`)."""
        return validate_input(self.model, data_version=self.contract_version, strict=strict)


def load_se_input(
    model: Any,
    *,
    contract_version: str = CONTRACT_VERSION,
) -> SEInput:
    """Wrap a model (already carrying measurements) into an input contract (``derived=None``).

    The input boundary is the ``.npz`` data file (see :func:`gridstate.contract.serialize.
    load_se_input_npz`), which restores the working layer + the precomputed
    ``DerivedInputs`` plans without any source-format layer. This helper wraps a ready model;
    the ``run`` steps that need derived inputs are skipped (``derived=None``).
    """
    return SEInput(model=model, derived=None, contract_version=contract_version)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


# name → OUTPUT schema of the table (single source — SEOutputSchema.tables()).
_SE_OUTPUT_BY_NAME = {t.name: t for t in SE_OUTPUT.tables()}


def _empty_output(table_name: str) -> np.ndarray:
    """Empty structured array of the output layer for table ``table_name``."""
    return np.empty(0, dtype=_SE_OUTPUT_BY_NAME[table_name].output_dtype())


@dataclass
class SEOutput:
    """SE output contract: results keyed by id (a separate results namespace).

    The tables correspond to :data:`gridstate.contract.tables.SE_OUTPUT`:
    ``nodes`` (V/δ + injections/imbalances/estimates), ``branches`` (flows/currents/
    losses/loading), ``measurements`` (estimates/residuals). Plus the solution scalars.

    Attributes:
        nodes/branches/measurements: structured arrays (id + OUTPUT columns).
        success/iterations/objective_value/algorithm: convergence metadata.
        v_pu/delta_rad: V in p.u. / angles (rad) — a duplicate for debugging/tests.
        contract_version: contract version of the result.
        result: the original ``SEResult`` — a back-compat escape hatch. Carries ``model``
            (working copy), ``chi2``, ``worst_residuals``, etc.
    """

    nodes: np.ndarray = field(default_factory=lambda: _empty_output("nodes"))
    branches: np.ndarray = field(default_factory=lambda: _empty_output("branches"))
    measurements: np.ndarray = field(default_factory=lambda: _empty_output("measurements"))
    success: bool = False
    iterations: int = 0
    objective_value: float = float("nan")
    algorithm: str = ""
    # Breakdown of success (see SEResult.convergence_status): for IPM
    # "kkt"/"completed"/"stalled"/"error", for WLS "converged"/"not_converged".
    convergence_status: str = ""
    message: str = ""
    v_pu: np.ndarray = field(default_factory=lambda: np.empty(0))
    delta_rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    contract_version: str = CONTRACT_VERSION
    result: SEResult | None = None

    @classmethod
    def from_result(cls, result: SEResult) -> SEOutput:
        """Build a :class:`SEOutput` from an ``SEResult`` (output tables + scalars)."""
        out = result.outputs
        return cls(
            nodes=out.nodes,
            branches=out.branches,
            measurements=out.measurements,
            success=bool(result.success),
            iterations=int(result.iterations),
            objective_value=float(result.objective_value),
            algorithm=str(result.algorithm),
            convergence_status=str(result.convergence_status),
            message=str(result.message),
            v_pu=result.v_pu,
            delta_rad=result.delta_rad,
            result=result,
        )

    def node(self, node_id: int) -> dict | None:
        """Output row of a node as a dict (or ``None``)."""
        return _row(self.nodes, node_id)

    def branch(self, branch_id: int) -> dict | None:
        """Output row of a branch as a dict (or ``None``)."""
        return _row(self.branches, branch_id)

    def measurement(self, meas_id: int) -> dict | None:
        """Output row of a measurement as a dict (or ``None``)."""
        return _row(self.measurements, meas_id)


def _row(arr: np.ndarray, obj_id: int) -> dict | None:
    if arr.size == 0:
        return None
    hits = np.where(arr["id"] == int(obj_id))[0]
    if hits.size == 0:
        return None
    row = arr[int(hits[0])]
    return {name: row[name].item() for name in (arr.dtype.names or ())}
