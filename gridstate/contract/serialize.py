"""Serialization of ``SEInput`` (contract + ``DerivedInputs``) to ``.npz`` — the input boundary.

**Purpose.** The only input boundary of gridstate is a **data file** (``.npz``)
prepared by the producer of the input and read by gridstate **without external
dependencies and without knowing the source format**.

Exactly what the kernel needs to run ``run(SEInput)`` is serialized:

* the ``SE_INPUT`` contract tables — ``nodes`` / ``branches`` / ``measurements`` /
  ``generators`` + the domain ``tap_steps`` / ``load_characteristics`` / ``shunts``
  (structured numpy arrays);
* :class:`~gridstate.contract.derived.DerivedInputs` — numeric plans (topology /
  telemetry / materialization / Vnom), the result of processing the source. Tap changer
  positions are applied through the ``tap_steps`` input table, not through ``DerivedInputs``.

The loader restores the working layer via :meth:`gridstate.working.Working.from_arrays`
(constructor from arrays) → ``run()`` executes without external dependencies.

**Derived inputs format.** Each ``DerivedInputs`` plan is stored as a typed
structured array under ``derived.<table>`` plus a JSON metadata document under
``derived.meta`` (see :mod:`gridstate.contract.derived_tables` for the tables,
their dtypes and the key vocabulary). Everything in the file is a plain numpy
array: it is read with ``allow_pickle=False`` and no code runs on load.

**Files of contract major 2** stored the plans as a pickled blob, and their
transformer from-side shunt fields had a different meaning (see
:mod:`gridstate.contract.version`). They are read only with
``load_se_input_npz(path, trusted=True)`` (unpickling can execute arbitrary
code); without it the loader raises :class:`UntrustedLegacyFormatError`. The
trusted path maps the plans onto the current vocabulary and warns
(:class:`LegacyContractWarning`); re-saving such an input with
:func:`save_se_input` writes the current format.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.contract._legacy import LEGACY_DERIVED_KEY, derived_from_legacy_blob
from gridstate.contract.derived_tables import (
    DERIVED_META_KEY,
    DERIVED_TABLE_PREFIX,
    DERIVED_TABLES,
    derived_from_tables,
    derived_to_tables,
    meta_from_json,
    meta_to_json,
)
from gridstate.contract.tables import SE_INPUT, SE_OUTPUT, io_dtype
from gridstate.contract.version import (
    CONTRACT_VERSION,
    LEGACY_MAJOR,
    ContractVersion,
    check_compatibility,
)


if TYPE_CHECKING:
    from gridstate.contract.containers import SEInput


_META_KEY = "__contract_version__"


class UntrustedLegacyFormatError(ValueError):
    """An ``.npz`` is a legacy (contract major 2) file.

    Its derived inputs are a pickled blob and its transformer shunt fields have the
    former meaning. Raised by :func:`load_se_input_npz` unless the caller passes
    ``trusted=True``.
    """


class LegacyContractWarning(UserWarning):
    """A legacy (contract major 2) file was read through the trusted path."""


_LEGACY_HINT = (
    "Unpickling can execute arbitrary code, so it is refused by default. If the file "
    "comes from a trusted source, load it with load_se_input_npz(path, trusted=True) and "
    "re-save it with save_se_input, or regenerate it with the current source tool."
)

_LEGACY_WARNING = (
    "{name!r} was built under contract {version} and is read through the legacy path. "
    "Since contract 3.0.0 the from-side shunt of a transformer with an off-nominal ratio "
    "(branches.conductance_from / susceptance_from) is connected at the from node, before "
    "the ideal transformer, and is no longer divided by |t|^2: an input that relied on "
    "the division is read with a different shunt. Regenerate it with the current source "
    "tool when that matters."
)

# Input table names in the order of the contract declaration (single source —
# SEInputSchema.tables()). The first four are the main contract tables
# (positional arguments of Working.from_arrays, with special handling below);
# the rest are domain numeric input tables: first-class (written/read
# under their own name) but optional (the source may not carry them:
# a missing one → an empty collection in Working.from_arrays).
_INPUT_TABLE_NAMES = tuple(t.name for t in SE_INPUT.tables())
_N_CORE_TABLES = 4
_CONTRACT_TABLES = _INPUT_TABLE_NAMES[:_N_CORE_TABLES]
_DOMAIN_TABLES = _INPUT_TABLE_NAMES[_N_CORE_TABLES:]


def _schema_map() -> dict[str, Any]:
    """name → ``TableSchema`` for the contract tables."""
    return {name: getattr(SE_INPUT, name) for name in _CONTRACT_TABLES}


def _derived_arrays(derived: Any) -> dict[str, np.ndarray]:
    """``DerivedInputs`` → ``.npz`` arrays (typed tables + JSON metadata); none for ``None``."""
    if derived is None:
        return {}
    tables, meta = derived_to_tables(derived)
    arrays = {DERIVED_TABLE_PREFIX + name: arr for name, arr in tables.items()}
    arrays[DERIVED_META_KEY] = np.asarray(meta_to_json(meta))
    return arrays


def _read_derived(npz: Any, files: set[str], path: str | Path, *, trusted: bool) -> Any:
    """Derived inputs of an open ``.npz``: typed tables, or the legacy blob if trusted."""
    if DERIVED_META_KEY in files:
        meta = meta_from_json(str(npz[DERIVED_META_KEY]))
        tables = {
            t.name: np.asarray(npz[DERIVED_TABLE_PREFIX + t.name])
            for t in DERIVED_TABLES
            if DERIVED_TABLE_PREFIX + t.name in files
        }
        return derived_from_tables(tables, meta)
    if LEGACY_DERIVED_KEY not in files:
        return None
    if not trusted:
        raise UntrustedLegacyFormatError(
            f"{Path(path).name!r} stores its derived inputs in the legacy pickled format "
            f"(contract major {LEGACY_MAJOR}). {_LEGACY_HINT}"
        )
    import pickle  # only on the explicit trusted path

    blob = pickle.loads(np.asarray(npz[LEGACY_DERIVED_KEY]).tobytes())
    return derived_from_legacy_blob(blob)


def _expand_to_io(arr: np.ndarray, in_schema: Any, out_schema: Any) -> np.ndarray:
    """Expand an array to the INPUT+OUTPUT dtype, filling the missing OUTPUT columns with zeros.

    Backward compatibility: if ``arr`` already contains OUTPUT fields (old format),
    they are copied as is.
    """
    # Full IO dtype — the single io_dtype builder (contract.tables).
    full_dt = io_dtype(in_schema, out_schema)
    result = np.zeros(len(arr), dtype=full_dt)
    for name in arr.dtype.names or ():
        if full_dt.names is not None and name in full_dt.names:
            result[name] = arr[name]
    return result


def save_se_input(se_input: SEInput, path: str | Path) -> Path:
    """Write ``SEInput`` (contract tables + ``DerivedInputs``) to an ``.npz``.

    **Only** the input contract columns are saved (``input_dtype()`` —
    KEY + INPUT + WORKING).  OUTPUT columns (``p_inj_calc``, ``estimated_si``,
    flows, etc.) are NOT written — they are the result of a ``run()``.
    The loader :func:`load_se_input_npz` adds empty OUTPUT columns for
    the pipeline.

    Args:
        se_input: SE input. ``se_input.model`` is the carrier of the contract tables
            (``Working`` or any object with the collections
            ``.nodes/.branches/.measurements/.generators`` + optional domain
            ``.tap_steps/.load_characteristics/.shunts`` that return ``to_numpy()``).
            ``se_input.derived`` — numeric plans (or ``None``).
        path: path of the output ``.npz`` (numpy appends the extension if missing).

    Returns:
        The actual path of the written file.
    """
    schema = _schema_map()
    model = se_input.model
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTRACT_TABLES:
        coll = getattr(model, name)
        full_arr = np.asarray(coll.to_numpy())
        # Generators have no OUTPUT — saved whole
        if name == "generators":
            arrays[name] = full_arr
            continue
        # Contract tables: input_dtype only (KEY + INPUT + WORKING)
        in_dt = schema[name].input_dtype()
        arr = np.empty(len(full_arr), dtype=in_dt)
        for col in in_dt.names:
            arr[col] = full_arr[col]
        arrays[name] = arr

    # Domain numeric tables — first-class, under their own name
    # (optional: the source may not carry them / may carry them empty).
    for name in _DOMAIN_TABLES:
        coll = getattr(model, name, None)
        if coll is not None and hasattr(coll, "to_numpy"):
            arr = np.asarray(coll.to_numpy())
            if len(arr) > 0:
                arrays[name] = arr

    arrays.update(_derived_arrays(se_input.derived))
    arrays[_META_KEY] = np.asarray(str(se_input.contract_version))

    out = Path(path)
    # mypy: **arrays statically collides with the keyword-only allow_pickle: bool in
    # savez; the arrays keys are data-array names only, allow_pickle is not among them.
    np.savez_compressed(out, **arrays)  # type: ignore[arg-type]
    # numpy appends .npz if there is no extension — return the actual name.
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def load_se_input_npz(path: str | Path, *, trusted: bool = False) -> SEInput:
    """Read an ``.npz`` (see :func:`save_se_input`) into ``SEInput``; no external dependencies.

    The working layer is built via :meth:`gridstate.working.Working.from_arrays`.
    The returned ``SEInput`` is ready for ``run(se_input)``: ``derived`` holds the
    restored numeric plans.

    The file is read with ``allow_pickle=False``; no code from the file runs.

    Args:
        path: the ``.npz`` file.
        trusted: also read a legacy file (contract major 2): its pickled derived
            inputs are unpickled and mapped onto the current vocabulary, a
            :class:`LegacyContractWarning` notes the changed transformer shunt
            semantics, and the returned input carries the current contract
            version. Unpickling can execute arbitrary code: pass ``True`` only
            for files from a trusted source, and re-save them with
            :func:`save_se_input`.

    Raises:
        UntrustedLegacyFormatError: the file is a legacy one (contract major 2 or
            pickled derived inputs) and ``trusted`` is ``False``.
        ValueError: incompatible contract version or malformed derived tables.
    """
    from gridstate.contract.containers import SEInput
    from gridstate.working import Working

    with np.load(path, allow_pickle=False) as npz:
        files = set(npz.files)
        contract_version = str(npz[_META_KEY]) if _META_KEY in files else None
        # Input boundary: the data version in the file must be compatible with the
        # library's built-in contract. Fail EARLY and EXPLICITLY (rather than silently
        # loading an incompatible schema that would fail deeper in the pipeline with an
        # obscure error).
        legacy = LEGACY_DERIVED_KEY in files and DERIVED_META_KEY not in files
        if contract_version is not None and check_compatibility(contract_version) is not None:
            if ContractVersion.parse(contract_version).major != LEGACY_MAJOR:
                raise ValueError(
                    f"Incompatible contract version in npz {Path(path).name!r}: "
                    f"the file was built under {contract_version}, the library expects "
                    f"{CONTRACT_VERSION} (same major, minor not newer). "
                    f"Regenerate the npz with the current source tool."
                )
            if not trusted:
                raise UntrustedLegacyFormatError(
                    f"{Path(path).name!r} was built under the legacy contract "
                    f"{contract_version}; the library expects {CONTRACT_VERSION}. "
                    f"{_LEGACY_HINT}"
                )
            legacy = True
        # Expand the contract tables to the INPUT+OUTPUT dtype
        # (add empty OUTPUT columns for the pipeline).
        nodes = _expand_to_io(np.asarray(npz["nodes"]), SE_INPUT.nodes, SE_OUTPUT.nodes)
        branches = _expand_to_io(np.asarray(npz["branches"]), SE_INPUT.branches, SE_OUTPUT.branches)
        measurements = _expand_to_io(
            np.asarray(npz["measurements"]), SE_INPUT.measurements, SE_OUTPUT.measurements
        )
        generators = np.asarray(npz["generators"])  # generators have no OUTPUT
        domain = {name: np.asarray(npz[name]) for name in _DOMAIN_TABLES if name in files}
        derived = _read_derived(npz, files, path, trusted=trusted)
        working = Working.from_arrays(
            nodes=nodes,
            branches=branches,
            measurements=measurements,
            generators=generators,
            **domain,
        )

    if legacy:
        warnings.warn(
            _LEGACY_WARNING.format(name=Path(path).name, version=contract_version or "2.x"),
            LegacyContractWarning,
            stacklevel=2,
        )
        # The tables and plans now follow the current contract (with the caveat above).
        contract_version = CONTRACT_VERSION
    kwargs: dict[str, Any] = {"model": working, "derived": derived}
    if contract_version is not None:
        kwargs["contract_version"] = contract_version
    return SEInput(**kwargs)
