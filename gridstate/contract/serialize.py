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

**Plan format (v0, provisional).** ``DerivedInputs`` carries dicts with tuple keys
(``telemetry_resolved``), so the plans are encoded with ``pickle`` in an object array inside
the ``.npz``. This is an internal Python format; a stable cross-tool format (a JSON schema
of the plans) is future work. The contract TABLES are stored as ordinary npz arrays and can be read by
any tool.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.contract.tables import SE_INPUT, SE_OUTPUT, io_dtype


if TYPE_CHECKING:
    from gridstate.contract.containers import SEInput


_DERIVED_KEY = "__derived_pickle__"
_META_KEY = "__contract_version__"

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


def _derived_to_blob(derived: Any) -> dict | None:
    """``DerivedInputs`` → serializable dict (numeric plans of the steps)."""
    if derived is None:
        return None
    return {
        "topology_resolved": derived.topology_resolved,
        "telemetry_resolved": derived.telemetry_resolved,
        "telemetry_arg_keys": derived.telemetry_arg_keys,
        "telemetry_total_args": derived.telemetry_total_args,
        "materialize_obs": derived.materialize_obs,
        "voltage_nominal": derived.voltage_nominal,
        "v_sigma2_scale": derived.v_sigma2_scale,
        "flow_sigma2_scale": derived.flow_sigma2_scale,
    }


def _blob_to_derived(blob: dict | None) -> Any:
    """Serialized dict → ``DerivedInputs`` (numeric plans of the steps)."""
    if blob is None:
        return None
    from gridstate.contract.derived import DerivedInputs

    return DerivedInputs(
        topology_resolved=blob["topology_resolved"],
        telemetry_resolved=blob["telemetry_resolved"],
        telemetry_arg_keys=blob["telemetry_arg_keys"],
        telemetry_total_args=blob["telemetry_total_args"],
        materialize_obs=blob["materialize_obs"],
        # .get(): old .npz files do not carry this plan (backward compatibility).
        voltage_nominal=blob["voltage_nominal"],
        v_sigma2_scale=blob.get("v_sigma2_scale"),
        flow_sigma2_scale=blob.get("flow_sigma2_scale"),
    )


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

    blob = _derived_to_blob(se_input.derived)
    arrays[_DERIVED_KEY] = np.frombuffer(pickle.dumps(blob), dtype=np.uint8)
    arrays[_META_KEY] = np.asarray(str(se_input.contract_version))

    out = Path(path)
    # mypy: **arrays statically collides with the keyword-only allow_pickle: bool in
    # savez; the arrays keys are data-array names only, allow_pickle is not among them.
    np.savez_compressed(out, **arrays)  # type: ignore[arg-type]
    # numpy appends .npz if there is no extension — return the actual name.
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def load_se_input_npz(path: str | Path) -> SEInput:
    """Read an ``.npz`` (see :func:`save_se_input`) into ``SEInput``; no external dependencies.

    The working layer is built via :meth:`gridstate.working.Working.from_arrays`.
    The returned ``SEInput`` is ready for ``run(se_input)``: ``derived`` holds the
    restored numeric plans.
    """
    from gridstate.contract.containers import SEInput
    from gridstate.contract.version import CONTRACT_VERSION, check_compatibility
    from gridstate.working import Working

    with np.load(path, allow_pickle=False) as npz:
        files = set(npz.files)
        # Expand the contract tables to the INPUT+OUTPUT dtype
        # (add empty OUTPUT columns for the pipeline).
        nodes = _expand_to_io(np.asarray(npz["nodes"]), SE_INPUT.nodes, SE_OUTPUT.nodes)
        branches = _expand_to_io(np.asarray(npz["branches"]), SE_INPUT.branches, SE_OUTPUT.branches)
        measurements = _expand_to_io(
            np.asarray(npz["measurements"]), SE_INPUT.measurements, SE_OUTPUT.measurements
        )
        generators = np.asarray(npz["generators"])  # generators have no OUTPUT
        domain = {name: np.asarray(npz[name]) for name in _DOMAIN_TABLES if name in files}
        working = Working.from_arrays(
            nodes=nodes,
            branches=branches,
            measurements=measurements,
            generators=generators,
            **domain,
        )
        blob = pickle.loads(bytes(npz[_DERIVED_KEY])) if _DERIVED_KEY in files else None
        contract_version = str(npz[_META_KEY]) if _META_KEY in files else None

    # Input boundary: the data version in the file must be compatible with the library's built-in
    # contract. Fail EARLY and EXPLICITLY (rather than silently loading an incompatible
    # schema that would fail deeper in the pipeline with an obscure error).
    if contract_version is not None and check_compatibility(contract_version) is not None:
        raise ValueError(
            f"Incompatible contract version in npz {Path(path).name!r}: "
            f"the file was built under {contract_version}, the library expects "
            f"{CONTRACT_VERSION} (majors must match). "
            f"Regenerate the npz with the current source tool."
        )

    derived = _blob_to_derived(blob)
    kwargs: dict[str, Any] = {"model": working, "derived": derived}
    if contract_version is not None:
        kwargs["contract_version"] = contract_version
    return SEInput(**kwargs)
