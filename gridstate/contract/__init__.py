"""Контракт данных оценки состояния (``SEInput`` / ``SEOutput``).

Выделенный, самодостаточный модуль gridstate, владеющий **схемой** входных и
выходных таблиц SE и **версией** контракта. Контракт делает зависимость SE от
*данных* явной, проверяемой и версионируемой — вместо неявной зависимости от
конкретного класса модели сети.

Состав:

* :mod:`gridstate.contract.tables` — декларация таблиц/колонок/ролей
  (:data:`SE_INPUT`, :data:`SE_OUTPUT`, :class:`Role`, :class:`ColumnSpec`,
  :class:`TableSchema`).
* :mod:`gridstate.contract.version` — :data:`CONTRACT_VERSION` + SemVer-логика
  совместимости (:class:`ContractVersion`).
* :mod:`gridstate.contract.validate` — :func:`validate_input` (валидация входа).
"""

from gridstate.contract.derived import DerivedInputs
from gridstate.contract.runtime import (
    SEInput,
    SEOutput,
    load_se_input,
    prepare_network,
    run,
)
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
