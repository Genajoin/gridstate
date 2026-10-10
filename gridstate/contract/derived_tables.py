"""Typed tables of the derived inputs (:class:`~gridstate.contract.derived.DerivedInputs`).

This module is the **public specification** of the derived inputs: the
vocabulary of their keys, the typed table every plan is stored as in the
``.npz`` contract file, and the JSON metadata that goes with them.

**Two forms of the same data.**

* *In memory* a :class:`~gridstate.contract.derived.DerivedInputs` holds each
  plan as plain Python containers (lists, dicts with tuple keys) that the
  pipeline steps apply directly. Their key vocabulary is defined below.
* *On disk* each plan is one structured numpy array (a "derived table") with
  fixed columns and dtypes, plus a small JSON metadata document. No object
  arrays and no ``pickle``: the file is read with ``allow_pickle=False`` and
  can be consumed by any tool that reads ``.npy`` arrays and JSON.

:func:`derived_to_tables` / :func:`derived_from_tables` convert between the two
forms and validate the vocabulary on the way (a key outside it raises
``ValueError``). The ``.npz`` reader and writer
(:mod:`gridstate.contract.serialize`) store the tables under
``derived.<table name>`` and the metadata under :data:`DERIVED_META_KEY`.

**Vocabulary** (string name in memory ↔ integer code on disk):

* topology object kinds :data:`TOPOLOGY_OBJECT_KINDS` — ``node`` / ``branch`` /
  ``generator`` (codes of ``measurements.object_type``);
* reasons why a status could not be derived :data:`SKIP_REASONS` —
  ``no_value`` / ``partial_inputs`` / ``evaluation_error``;
* telemetry kinds :data:`TELEMETRY_KINDS` — node voltage ``v``, branch flows
  ``p_from`` / ``p_to`` / ``q_from`` / ``q_to`` (:data:`FLOW_KINDS`), node
  injection components ``p_gen`` / ``p_load`` / ``q_gen`` / ``q_load``
  (:data:`INJECTION_KINDS`). An injection component may carry a component
  number, ``"<kind>:<n>"`` (``n >= 0``, e.g. ``"p_gen:3"`` for one generating
  unit of a node), so that several components of one kind on one node stay
  distinct; the components of a node are summed into its injection
  measurement;
* observed node quantities :data:`MATERIALIZE_QUANTITIES` — ``load_p`` /
  ``load_q`` / ``gen_p`` / ``gen_q``.

**Fields and tables** (one table per plan; a plan that is ``None`` writes no
table and is absent from ``fields`` in the metadata):

=====================  =====================  ======================================
``DerivedInputs``      table                  row
=====================  =====================  ======================================
``topology_resolved``  ``topology``           one item of the status plan
``telemetry_resolved`` ``telemetry``          one derived telemetry value
``telemetry_arg_keys`` ``telemetry_order``    one key, in application order
``materialize_obs``    ``materialize``        one observed node quantity
``voltage_nominal``    ``voltage_nominal``    one node nominal voltage
``v_sigma2_scale``     ``v_sigma2_scale``     one node V variance factor
``flow_sigma2_scale``  ``flow_sigma2_scale``  one branch flow variance factor
``telemetry_total_args``  (metadata)          integer, statistics only
=====================  =====================  ======================================

The column dtypes and meanings are declared in :data:`DERIVED_TABLES` (the
same :class:`~gridstate.contract.tables.TableSchema` machinery as the contract
tables). Every table has shape ``(n_rows,)``.

**Metadata** (:data:`DERIVED_META_KEY`, a JSON object stored as a 0-d unicode
array)::

    {
      "fields": ["topology_resolved", ...],   # plans that are not None
      "telemetry_total_args": 0,              # DerivedInputs.telemetry_total_args
      "materialize_quantities": ["load_p", ...]  # keys of materialize_obs, in order
    }

A file without the metadata key carries no derived inputs (``derived=None``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.constants import MeasurementObjectType
from gridstate.contract.tables import ColumnSpec, Role, TableSchema


if TYPE_CHECKING:
    from gridstate.contract.derived import DerivedInputs


__all__ = [
    "COMPONENT_SEPARATOR",
    "DERIVED_META_KEY",
    "DERIVED_TABLES",
    "DERIVED_TABLE_PREFIX",
    "FIELD_TABLES",
    "FLOW_KINDS",
    "INJECTION_KINDS",
    "KIND_P_FROM",
    "KIND_P_GEN",
    "KIND_P_LOAD",
    "KIND_P_TO",
    "KIND_Q_FROM",
    "KIND_Q_GEN",
    "KIND_Q_LOAD",
    "KIND_Q_TO",
    "KIND_V",
    "MATERIALIZE_QUANTITIES",
    "NO_COMPONENT",
    "NO_SKIP_REASON",
    "SKIP_REASONS",
    "SOURCE_GUID_WIDTH",
    "TELEMETRY_KINDS",
    "TOPOLOGY_OBJECT_KINDS",
    "derived_from_tables",
    "derived_to_tables",
    "join_kind",
    "kind_base",
    "meta_from_json",
    "meta_to_json",
    "split_kind",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Object kinds of the topology plan → code (``measurements.object_type`` codes).
TOPOLOGY_OBJECT_KINDS: dict[str, int] = {
    "node": int(MeasurementObjectType.NODE),
    "branch": int(MeasurementObjectType.BRANCH),
    "generator": int(MeasurementObjectType.GENERATOR),
}

#: Why the status of an object could not be derived → code (``0`` = derived).
SKIP_REASONS: dict[str, int] = {
    "no_value": 1,  # no source value available
    "partial_inputs": 2,  # only some of the source values available
    "evaluation_error": 3,  # the source values could not be combined
}
NO_SKIP_REASON = 0

#: Branch flow kinds (from / to end of the branch, active / reactive).
#: Telemetry kind names — the single source of the key strings (also used by
#: :mod:`gridstate.telemetry.kinds` and the telemetry kernels).
KIND_V = "v"  # node voltage magnitude
KIND_P_FROM = "p_from"  # branch active power flow at the from end
KIND_P_TO = "p_to"  # branch active power flow at the to end
KIND_Q_FROM = "q_from"  # branch reactive power flow at the from end
KIND_Q_TO = "q_to"  # branch reactive power flow at the to end
KIND_P_GEN = "p_gen"  # node active generation component
KIND_P_LOAD = "p_load"  # node active load component
KIND_Q_GEN = "q_gen"  # node reactive generation component
KIND_Q_LOAD = "q_load"  # node reactive load component

#: Branch flow kinds (from / to end of the branch, active / reactive).
FLOW_KINDS: tuple[str, ...] = (KIND_P_FROM, KIND_P_TO, KIND_Q_FROM, KIND_Q_TO)
#: Node injection components (generation counts +, load counts −).
INJECTION_KINDS: tuple[str, ...] = (KIND_P_GEN, KIND_P_LOAD, KIND_Q_GEN, KIND_Q_LOAD)
#: Telemetry kinds → code.
TELEMETRY_KINDS: dict[str, int] = {
    KIND_V: 0,
    **{k: 1 + i for i, k in enumerate(FLOW_KINDS)},
    **{k: 1 + len(FLOW_KINDS) + i for i, k in enumerate(INJECTION_KINDS)},
}
#: Separator of the component number in an injection kind (``"p_gen:3"``).
COMPONENT_SEPARATOR = ":"
#: Component column value of a kind without a component number.
NO_COMPONENT = -1

#: Observed node quantities of the materialization plan → code.
MATERIALIZE_QUANTITIES: dict[str, int] = {"load_p": 0, "load_q": 1, "gen_p": 2, "gen_q": 3}


def kind_base(kind: str) -> str:
    """Kind without its component number (``"p_gen:3"`` → ``"p_gen"``); never raises."""
    return kind.partition(COMPONENT_SEPARATOR)[0]


def split_kind(kind: str) -> tuple[str, int]:
    """Validate a telemetry kind and split it into ``(base, component)``.

    ``"p_gen:3"`` → ``("p_gen", 3)``; ``"p_from"`` → ``("p_from", NO_COMPONENT)``.

    Raises:
        ValueError: unknown base kind, a component number on a kind other than
            an injection component, or a malformed component number.
    """
    base, sep, comp = str(kind).partition(COMPONENT_SEPARATOR)
    if base not in TELEMETRY_KINDS:
        raise ValueError(
            f"unknown telemetry kind {kind!r}; expected one of {sorted(TELEMETRY_KINDS)}"
        )
    if not sep:
        return base, NO_COMPONENT
    if base not in INJECTION_KINDS:
        raise ValueError(
            f"telemetry kind {kind!r}: only injection components {INJECTION_KINDS} "
            f"take a component number"
        )
    if not comp.isdigit() or not comp.isascii():
        raise ValueError(f"telemetry kind {kind!r}: the component number must be an integer >= 0")
    return base, int(comp)


def join_kind(base: str, component: int) -> str:
    """Inverse of :func:`split_kind` (``("p_gen", 3)`` → ``"p_gen:3"``)."""
    return base if component == NO_COMPONENT else f"{base}{COMPONENT_SEPARATOR}{component}"


# ---------------------------------------------------------------------------
# Table declarations
# ---------------------------------------------------------------------------


def _col(name: str, dtype: str, doc: str) -> ColumnSpec:
    return ColumnSpec(name, dtype, Role.INPUT, doc=doc)


#: Source identifier width (unicode characters). Longer values are rejected.
SOURCE_GUID_WIDTH = 64

TOPOLOGY = TableSchema(
    name="topology",
    key=("object_type", "object_id"),
    doc=(
        "Status plan (DerivedInputs.topology_resolved), in application order. In memory: "
        "list of (object_kind, object_id, status | None, skip_reason | None)."
    ),
    columns=(
        _col(
            "object_type",
            "i1",
            "Object kind code (TOPOLOGY_OBJECT_KINDS: 0 node/1 branch/2 generator).",
        ),
        _col("object_id", "i4", "Id of the object in its contract table."),
        _col("has_status", "bool", "Whether a status was derived (False: in memory status=None)."),
        _col("status", "bool", "Derived in-service status (meaningful when has_status)."),
        _col("skip_reason", "i1", "SKIP_REASONS code when no status was derived; 0 otherwise."),
    ),
)

TELEMETRY = TableSchema(
    name="telemetry",
    key=("object_id", "kind", "component"),
    doc=(
        "Derived telemetry values (DerivedInputs.telemetry_resolved), in mapping order. In "
        "memory: {(object_id, kind): (value | None, n_inputs, source_guid, quality)}."
    ),
    columns=(
        _col("object_id", "i4", "Node id (v and injection kinds) or branch id (flow kinds)."),
        _col("kind", "i1", "Telemetry kind code (TELEMETRY_KINDS)."),
        _col("component", "i4", "Component number of an injection kind; -1 for none."),
        _col("has_value", "bool", "Whether a value was derived (False: in memory value=None)."),
        _col("value", "f8", "Value in engineering units (kV, MW, Mvar), sign applied."),
        _col("n_inputs", "i4", "Number of source values used (0 with no value: none available)."),
        _col("source_guid", f"U{SOURCE_GUID_WIDTH}", "Identifier of the first source value."),
        _col("quality", "i1", "Quality class (MeasurementQuality: 0 good/1 questionable/2 bad)."),
    ),
)

TELEMETRY_ORDER = TableSchema(
    name="telemetry_order",
    key=("object_id", "kind", "component"),
    doc=(
        "Application order of the telemetry values (DerivedInputs.telemetry_arg_keys). In "
        "memory: list of (object_id, kind)."
    ),
    columns=(
        _col("object_id", "i4", "Object id (as in the telemetry table)."),
        _col("kind", "i1", "Telemetry kind code (TELEMETRY_KINDS)."),
        _col("component", "i4", "Component number of an injection kind; -1 for none."),
    ),
)

MATERIALIZE = TableSchema(
    name="materialize",
    key=("quantity", "node_id"),
    doc=(
        "Observed node injections (DerivedInputs.materialize_obs). In memory: "
        "{quantity: {node_id: value}}; the quantities present are listed in the metadata."
    ),
    columns=(
        _col(
            "quantity", "i1", "Quantity code (MATERIALIZE_QUANTITIES: load_p/load_q/gen_p/gen_q)."
        ),
        _col("node_id", "i4", "Node id."),
        _col("value", "f8", "Observed value, MW or Mvar."),
    ),
)

VOLTAGE_NOMINAL = TableSchema(
    name="voltage_nominal",
    key=("node_id",),
    doc="Nominal voltage of nodes (DerivedInputs.voltage_nominal). In memory: {node_id: kV}.",
    columns=(
        _col("node_id", "i4", "Node id."),
        _col("value", "f8", "Nominal voltage, kV (applied where nodes.voltage_nominal is 0)."),
    ),
)

V_SIGMA2_SCALE = TableSchema(
    name="v_sigma2_scale",
    key=("node_id",),
    doc=(
        "Variance factor of node voltage measurements (DerivedInputs.v_sigma2_scale). In "
        "memory: {node_id: factor}."
    ),
    columns=(
        _col("node_id", "i4", "Node id."),
        _col("factor", "f8", "Multiplier of the measurement variance (<1 more trust, >1 less)."),
    ),
)

FLOW_SIGMA2_SCALE = TableSchema(
    name="flow_sigma2_scale",
    key=("branch_id", "kind"),
    doc=(
        "Variance factor of branch flow measurements (DerivedInputs.flow_sigma2_scale). In "
        "memory: {(branch_id, flow_kind): factor}."
    ),
    columns=(
        _col("branch_id", "i4", "Branch id."),
        _col("kind", "i1", "Flow kind code (TELEMETRY_KINDS restricted to FLOW_KINDS)."),
        _col("factor", "f8", "Multiplier of the measurement variance (<1 more trust, >1 less)."),
    ),
)

#: All derived tables, in declaration order.
DERIVED_TABLES: tuple[TableSchema, ...] = (
    TOPOLOGY,
    TELEMETRY,
    TELEMETRY_ORDER,
    MATERIALIZE,
    VOLTAGE_NOMINAL,
    V_SIGMA2_SCALE,
    FLOW_SIGMA2_SCALE,
)

#: ``DerivedInputs`` plan field → name of the table that stores it.
FIELD_TABLES: dict[str, str] = {
    "topology_resolved": TOPOLOGY.name,
    "telemetry_resolved": TELEMETRY.name,
    "telemetry_arg_keys": TELEMETRY_ORDER.name,
    "materialize_obs": MATERIALIZE.name,
    "voltage_nominal": VOLTAGE_NOMINAL.name,
    "v_sigma2_scale": V_SIGMA2_SCALE.name,
    "flow_sigma2_scale": FLOW_SIGMA2_SCALE.name,
}

#: Prefix of the derived table keys in the ``.npz`` file (``derived.topology`` …).
DERIVED_TABLE_PREFIX = "derived."
#: Key of the JSON metadata of the derived inputs in the ``.npz`` file.
DERIVED_META_KEY = "derived.meta"

_SCHEMAS: dict[str, TableSchema] = {t.name: t for t in DERIVED_TABLES}
_TOPOLOGY_KIND_NAMES = {v: k for k, v in TOPOLOGY_OBJECT_KINDS.items()}
_SKIP_REASON_NAMES = {v: k for k, v in SKIP_REASONS.items()}
_TELEMETRY_KIND_NAMES = {v: k for k, v in TELEMETRY_KINDS.items()}
_QUANTITY_NAMES = {v: k for k, v in MATERIALIZE_QUANTITIES.items()}


def _code(vocabulary: Mapping[str, int], name: Any, what: str) -> int:
    try:
        return vocabulary[name]
    except (KeyError, TypeError):
        raise ValueError(f"unknown {what} {name!r}; expected one of {sorted(vocabulary)}") from None


def _name(names: Mapping[int, str], code: int, what: str) -> str:
    try:
        return names[code]
    except KeyError:
        raise ValueError(f"unknown {what} code {code}; expected one of {sorted(names)}") from None


def _table(name: str, rows: Iterable[tuple]) -> np.ndarray:
    return np.array(list(rows), dtype=_SCHEMAS[name].numpy_dtype())


def _flow_code(kind: str) -> int:
    if kind not in FLOW_KINDS:
        raise ValueError(f"unknown flow kind {kind!r}; expected one of {list(FLOW_KINDS)}")
    return TELEMETRY_KINDS[kind]


def _guid(guid: Any) -> str:
    text = str(guid)
    if len(text) > SOURCE_GUID_WIDTH:
        raise ValueError(
            f"source identifier longer than {SOURCE_GUID_WIDTH} characters: {text[:80]!r}"
        )
    return text


# ---------------------------------------------------------------------------
# In memory -> tables
# ---------------------------------------------------------------------------


def _topology_rows(plan: Iterable[tuple]) -> Iterable[tuple]:
    for kind, object_id, status, reason in plan:
        if status is None and reason is None:
            raise ValueError(
                f"topology item ({kind!r}, {object_id}): status None needs a skip reason"
            )
        yield (
            _code(TOPOLOGY_OBJECT_KINDS, kind, "topology object kind"),
            int(object_id),
            status is not None,
            bool(status) if status is not None else False,
            NO_SKIP_REASON if reason is None else _code(SKIP_REASONS, reason, "skip reason"),
        )


def _key_columns(object_id: Any, kind: str) -> tuple[int, int, int]:
    base, component = split_kind(kind)
    return int(object_id), TELEMETRY_KINDS[base], component


def _telemetry_rows(values: Mapping[tuple[int, str], tuple]) -> Iterable[tuple]:
    for (object_id, kind), (value, n_inputs, guid, quality) in values.items():
        yield (
            *_key_columns(object_id, kind),
            value is not None,
            float(value) if value is not None else 0.0,
            int(n_inputs),
            _guid(guid),
            int(quality),
        )


def _materialize_rows(obs: Mapping[str, Mapping[int, float]]) -> Iterable[tuple]:
    for quantity, by_node in obs.items():
        code = _code(MATERIALIZE_QUANTITIES, quantity, "materialize quantity")
        for node_id, value in by_node.items():
            yield code, int(node_id), float(value)


def derived_to_tables(derived: DerivedInputs) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """``DerivedInputs`` → ``({table name: structured array}, metadata)``.

    Only the plans that are not ``None`` produce a table. The vocabulary is
    validated: an unknown kind, object kind, skip reason or quantity raises
    ``ValueError`` (nothing outside the specification reaches the file).
    """
    tables: dict[str, np.ndarray] = {}
    present: list[str] = []

    def put(field: str, rows: Iterable[tuple]) -> None:
        tables[FIELD_TABLES[field]] = _table(FIELD_TABLES[field], rows)
        present.append(field)

    if derived.topology_resolved is not None:
        put("topology_resolved", _topology_rows(derived.topology_resolved))
    if derived.telemetry_resolved is not None:
        put("telemetry_resolved", _telemetry_rows(derived.telemetry_resolved))
    if derived.telemetry_arg_keys is not None:
        put("telemetry_arg_keys", (_key_columns(o, k) for o, k in derived.telemetry_arg_keys))
    quantities: list[str] = []
    if derived.materialize_obs is not None:
        quantities = [str(q) for q in derived.materialize_obs]
        put("materialize_obs", _materialize_rows(derived.materialize_obs))
    if derived.voltage_nominal is not None:
        put("voltage_nominal", ((int(n), float(v)) for n, v in derived.voltage_nominal.items()))
    if derived.v_sigma2_scale is not None:
        put("v_sigma2_scale", ((int(n), float(f)) for n, f in derived.v_sigma2_scale.items()))
    if derived.flow_sigma2_scale is not None:
        put(
            "flow_sigma2_scale",
            ((int(b), _flow_code(k), float(f)) for (b, k), f in derived.flow_sigma2_scale.items()),
        )
    meta = {
        "fields": present,
        "telemetry_total_args": int(derived.telemetry_total_args),
        "materialize_quantities": quantities,
    }
    return tables, meta


# ---------------------------------------------------------------------------
# Tables -> in memory
# ---------------------------------------------------------------------------


def _typed(name: str, arr: np.ndarray) -> np.ndarray:
    """Check the columns of a stored table and cast it to the declared dtype (safe casts only)."""
    want = _SCHEMAS[name].numpy_dtype()
    arr = np.asarray(arr)
    if arr.ndim != 1 or set(arr.dtype.names or ()) != set(want.names or ()):
        raise ValueError(
            f"derived table {name!r}: expected a 1-d table with columns {list(want.names or ())}, "
            f"got shape {arr.shape} with columns {list(arr.dtype.names or ())}"
        )
    out = np.empty(len(arr), dtype=want)
    for col in want.names or ():
        if not np.can_cast(arr.dtype[col], want[col], casting="safe"):
            raise ValueError(
                f"derived table {name!r}: column {col!r} has dtype {arr.dtype[col]}, "
                f"expected {want[col]}"
            )
        out[col] = arr[col]
    return out


def _unique(name: str, keys: list) -> None:
    if len(set(keys)) != len(keys):
        raise ValueError(f"derived table {name!r}: duplicate keys")


def _topology_plan(arr: np.ndarray) -> list[tuple[str, int, bool | None, str | None]]:
    plan: list[tuple[str, int, bool | None, str | None]] = []
    for code, object_id, has_status, status, reason in arr.tolist():
        if not has_status and reason == NO_SKIP_REASON:
            raise ValueError(
                f"derived table 'topology': object {object_id} without status or reason"
            )
        plan.append(
            (
                _name(_TOPOLOGY_KIND_NAMES, code, "topology object kind"),
                object_id,
                status if has_status else None,
                None
                if reason == NO_SKIP_REASON
                else _name(_SKIP_REASON_NAMES, reason, "skip reason"),
            )
        )
    return plan


def _kind_name(code: int, component: int) -> str:
    base = _name(_TELEMETRY_KIND_NAMES, code, "telemetry kind")
    if component != NO_COMPONENT and (component < 0 or base not in INJECTION_KINDS):
        raise ValueError(f"telemetry kind {base!r}: invalid component number {component}")
    return join_kind(base, component)


def _telemetry_values(
    arr: np.ndarray,
) -> dict[tuple[int, str], tuple[float | None, int, str, int]]:
    out: dict[tuple[int, str], tuple[float | None, int, str, int]] = {}
    for object_id, code, comp, has_value, value, n_inputs, guid, quality in arr.tolist():
        out[(object_id, _kind_name(code, comp))] = (
            value if has_value else None,
            n_inputs,
            guid,
            quality,
        )
    if len(out) != len(arr):
        raise ValueError("derived table 'telemetry': duplicate keys")
    return out


def _telemetry_order(arr: np.ndarray) -> list[tuple[int, str]]:
    keys = [(object_id, _kind_name(code, comp)) for object_id, code, comp in arr.tolist()]
    _unique("telemetry_order", keys)
    return keys


def _materialize_obs(arr: np.ndarray, quantities: list[str]) -> dict[str, dict[int, float]]:
    obs: dict[str, dict[int, float]] = {}
    for quantity in quantities:
        _code(MATERIALIZE_QUANTITIES, quantity, "materialize quantity")
        obs[quantity] = {}
    for code, node_id, value in arr.tolist():
        quantity = _name(_QUANTITY_NAMES, code, "materialize quantity")
        if quantity not in obs:
            raise ValueError(f"derived table 'materialize': quantity {quantity!r} not in metadata")
        if node_id in obs[quantity]:
            raise ValueError("derived table 'materialize': duplicate keys")
        obs[quantity][node_id] = value
    return obs


def _by_node(name: str, arr: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = dict(arr.tolist())
    if len(out) != len(arr):
        raise ValueError(f"derived table {name!r}: duplicate keys")
    return out


def _flow_scale(arr: np.ndarray) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for branch_id, code, factor in arr.tolist():
        kind = _name(_TELEMETRY_KIND_NAMES, code, "flow kind")
        _flow_code(kind)
        out[(branch_id, kind)] = factor
    if len(out) != len(arr):
        raise ValueError("derived table 'flow_sigma2_scale': duplicate keys")
    return out


def derived_from_tables(tables: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> DerivedInputs:
    """``({table name: structured array}, metadata)`` → ``DerivedInputs``.

    Inverse of :func:`derived_to_tables`. Every plan listed in
    ``meta["fields"]`` must have its table; the tables are checked against
    :data:`DERIVED_TABLES` (columns, safe dtype casts, known codes, unique keys).

    Raises:
        ValueError: malformed metadata or tables.
    """
    from gridstate.contract.derived import DerivedInputs

    fields = meta.get("fields")
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        raise ValueError("derived metadata: 'fields' must be a list of field names")
    unknown = sorted(set(fields) - set(FIELD_TABLES))
    if unknown:
        raise ValueError(f"derived metadata: unknown fields {unknown}")
    total = meta.get("telemetry_total_args", 0)
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("derived metadata: 'telemetry_total_args' must be an integer")
    quantities = meta.get("materialize_quantities", [])
    if not isinstance(quantities, list) or not all(isinstance(q, str) for q in quantities):
        raise ValueError("derived metadata: 'materialize_quantities' must be a list of names")

    def table(field: str) -> np.ndarray | None:
        if field not in fields:
            return None
        name = FIELD_TABLES[field]
        if name not in tables:
            raise ValueError(f"derived metadata lists {field!r} but table {name!r} is missing")
        return _typed(name, tables[name])

    derived = DerivedInputs(telemetry_total_args=total)
    if (arr := table("topology_resolved")) is not None:
        derived.topology_resolved = _topology_plan(arr)
    if (arr := table("telemetry_resolved")) is not None:
        derived.telemetry_resolved = _telemetry_values(arr)
    if (arr := table("telemetry_arg_keys")) is not None:
        derived.telemetry_arg_keys = _telemetry_order(arr)
    if (arr := table("materialize_obs")) is not None:
        derived.materialize_obs = _materialize_obs(arr, quantities)
    if (arr := table("voltage_nominal")) is not None:
        derived.voltage_nominal = _by_node("voltage_nominal", arr)
    if (arr := table("v_sigma2_scale")) is not None:
        derived.v_sigma2_scale = _by_node("v_sigma2_scale", arr)
    if (arr := table("flow_sigma2_scale")) is not None:
        derived.flow_sigma2_scale = _flow_scale(arr)
    return derived


def meta_to_json(meta: Mapping[str, Any]) -> str:
    """Metadata → canonical JSON text (sorted keys)."""
    return json.dumps(dict(meta), sort_keys=True)


def meta_from_json(text: str) -> dict[str, Any]:
    """JSON text → metadata object.

    Raises:
        ValueError: the text is not a JSON object.
    """
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"derived metadata is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError("derived metadata must be a JSON object")
    return meta
