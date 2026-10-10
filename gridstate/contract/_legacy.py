"""Reader of the derived inputs of contract files older than 2.1.0 (trusted files only).

Up to contract 2.0.x the ``.npz`` writer stored ``DerivedInputs`` as one pickled
blob under :data:`LEGACY_DERIVED_KEY`, with the key vocabulary of the producer
of the input. Unpickling runs arbitrary code, so
:func:`gridstate.contract.serialize.load_se_input_npz` unpickles the blob only
when the caller passes ``trusted=True``. This module then maps the old
vocabulary onto the current one (:mod:`gridstate.contract.derived_tables`).

The mapping is kept here only to read old files; the writer always produces
the typed tables. Items the core never applied (topology objects other than
nodes / branches / generators, telemetry kinds outside the vocabulary) are
dropped: they only fed the "no object" / "unsupported kind" counters.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from gridstate.contract import derived_tables as dt


if TYPE_CHECKING:
    from gridstate.contract.derived import DerivedInputs


LEGACY_DERIVED_KEY = "__derived_pickle__"

_TELEMETRY_KINDS = {
    "U": dt.KIND_V,
    "PBEG": dt.KIND_P_FROM,
    "PEND": dt.KIND_P_TO,
    "QBEG": dt.KIND_Q_FROM,
    "QEND": dt.KIND_Q_TO,
    "PG": dt.KIND_P_GEN,
    "PN": dt.KIND_P_LOAD,
    "QG": dt.KIND_Q_GEN,
    "QN": dt.KIND_Q_LOAD,
}
# Per-unit generation components: ``PG_G<n>`` / ``QG_G<n>`` → ``p_gen:<n>`` / ``q_gen:<n>``.
_UNIT_KIND = re.compile(r"^(PG|QG)_G(\d+)$")
_OBJECT_KINDS = {"NODE": "node", "LINE": "branch", "GENERATOR": "generator"}
_SKIP_REASONS = {
    "skipped_no_value": "no_value",
    "skipped_partial_args": "partial_inputs",
    "skipped_formula_error": "evaluation_error",
}

_log = logging.getLogger(__name__)


def legacy_kind(kind: Any) -> str | None:
    """Old telemetry kind → current kind (``None`` if it has no counterpart)."""
    if not isinstance(kind, str):
        return None
    if kind in _TELEMETRY_KINDS:
        return _TELEMETRY_KINDS[kind]
    match = _UNIT_KIND.match(kind)
    if match is None:
        return None
    return dt.join_kind(_TELEMETRY_KINDS[match.group(1)], int(match.group(2)))


def _topology(plan: list | None) -> tuple[list | None, int]:
    if plan is None:
        return None, 0
    out = []
    for tag, object_id, status, reason in plan:
        kind = _OBJECT_KINDS.get(tag)
        if kind is None:
            continue
        out.append((kind, object_id, status, None if reason is None else _SKIP_REASONS[reason]))
    return out, len(plan) - len(out)


def _telemetry(
    values: dict | None, order: list | None
) -> tuple[dict | None, list | None, set[str]]:
    dropped: set[str] = set()

    def key(k: tuple) -> tuple | None:
        new = legacy_kind(k[1])
        if new is None:
            dropped.add(str(k[1]))
            return None
        return (k[0], new)

    new_values = None
    if values is not None:
        new_values = {nk: v for k, v in values.items() if (nk := key(k)) is not None}
    new_order = None
    if order is not None:
        new_order = [nk for k in order if (nk := key(k)) is not None]
    return new_values, new_order, dropped


def _flow_scale(scale: dict | None) -> dict | None:
    if scale is None:
        return None
    out = {}
    for (branch_id, kind), factor in scale.items():
        new = legacy_kind(kind)
        if new is not None:
            out[(branch_id, new)] = factor
    return out


def derived_from_legacy_blob(blob: dict | None) -> DerivedInputs | None:
    """Unpickled legacy blob → ``DerivedInputs`` in the current vocabulary.

    Raises:
        ValueError: the blob is not a mapping of plans, or a skip reason is unknown.
    """
    if blob is None:
        return None
    if not isinstance(blob, dict):
        raise ValueError(f"legacy derived inputs: expected a dict, got {type(blob).__name__}")
    from gridstate.contract.derived import DerivedInputs

    try:
        topology, n_topology_dropped = _topology(blob.get("topology_resolved"))
    except KeyError as exc:
        raise ValueError(f"legacy derived inputs: unknown skip reason {exc}") from None
    values, order, dropped = _telemetry(
        blob.get("telemetry_resolved"), blob.get("telemetry_arg_keys")
    )
    if n_topology_dropped:
        _log.info(
            "legacy derived inputs: dropped %d topology item(s) of object kinds the core "
            "does not apply",
            n_topology_dropped,
        )
    if dropped:
        _log.warning(
            "legacy derived inputs: dropped telemetry kinds without a counterpart: %s",
            sorted(dropped),
        )
    return DerivedInputs(
        topology_resolved=topology,
        telemetry_resolved=values,
        telemetry_arg_keys=order,
        telemetry_total_args=int(blob.get("telemetry_total_args", 0)),
        materialize_obs=blob.get("materialize_obs"),
        voltage_nominal=blob.get("voltage_nominal"),
        v_sigma2_scale=blob.get("v_sigma2_scale"),
        flow_sigma2_scale=_flow_scale(blob.get("flow_sigma2_scale")),
    )
