"""State estimation data contract (``SEInput`` / ``SEOutput``).

A dedicated, self-contained gridstate module that owns the **schema** of the SE input and
output tables and the contract **version**. The contract makes the dependence of SE on
*data* explicit, verifiable and versioned, instead of an implicit dependence on a
particular network-model class.

Contents:

* :mod:`gridstate.contract.tables` — declaration of tables/columns/roles
  (:data:`SE_INPUT`, :data:`SE_OUTPUT`, :class:`Role`, :class:`ColumnSpec`,
  :class:`TableSchema`).
* :mod:`gridstate.contract.version` — :data:`CONTRACT_VERSION` + SemVer
  compatibility logic (:class:`ContractVersion`).
* :mod:`gridstate.contract.validate` — :func:`validate_input` (input validation).
"""

from gridstate.contract.containers import SEInput, SEOutput, load_se_input
from gridstate.contract.derived import DerivedInputs
from gridstate.contract.tables import (
    BRANCHES,
    BRANCHES_OUTPUT,
    GENERATORS,
    LOAD_CHARACTERISTICS,
    MEASUREMENTS,
    MEASUREMENTS_OUTPUT,
    NODES,
    NODES_OUTPUT,
    SE_INPUT,
    SE_OUTPUT,
    SHUNTS,
    TAP_STEPS,
    ColumnSpec,
    Role,
    SEInputSchema,
    SEOutputSchema,
    TableSchema,
    io_dtype,
)
from gridstate.contract.validate import (
    ContractValidationError,
    ValidationIssue,
    ValidationReport,
    validate_input,
)
from gridstate.contract.version import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_KEY,
    ContractVersion,
    current_version,
    is_data_compatible,
)

# The SE facade lives in gridstate.runtime (orchestration does not belong to
# the contract). Its functions are re-exported here as attributes of this
# package: callers that do ``from gridstate.contract import run`` at call time
# resolve the name here, so patching ``gridstate.contract.run`` reaches them.
from gridstate.runtime import prepare_network, run


__all__ = [
    "BRANCHES",
    "BRANCHES_OUTPUT",
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_KEY",
    "GENERATORS",
    "LOAD_CHARACTERISTICS",
    "MEASUREMENTS",
    "MEASUREMENTS_OUTPUT",
    "NODES",
    "NODES_OUTPUT",
    "SE_INPUT",
    "SE_OUTPUT",
    "SHUNTS",
    "TAP_STEPS",
    "ColumnSpec",
    "ContractValidationError",
    "ContractVersion",
    "DerivedInputs",
    "Role",
    "SEInput",
    "SEInputSchema",
    "SEOutput",
    "SEOutputSchema",
    "TableSchema",
    "ValidationIssue",
    "ValidationReport",
    "current_version",
    "io_dtype",
    "is_data_compatible",
    "load_se_input",
    "prepare_network",
    "run",
    "validate_input",
]
