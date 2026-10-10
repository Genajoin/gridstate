"""``DerivedInputs`` — numeric plans of the SE inputs, precomputed outside the ``run()`` kernel.

A container for the result of processing a data source: topology/telemetry/
materialization/Vnom are already reduced to numbers (by an external data producer) and
serialized into a boundary file (``.npz``, see :mod:`gridstate.contract.serialize`).
The SE kernel applies these plans through the contract kernels (``apply_*_resolved``) without knowing
which source format they came from.

The key vocabulary of the plans and their typed on-disk tables are specified in
:mod:`gridstate.contract.derived_tables`.
"""

from __future__ import annotations

from dataclasses import dataclass


#: Topology plan item: ``(object_kind, object_id, status | None, skip_reason | None)``.
#: ``object_kind`` ∈ ``node``/``branch``/``generator``; ``skip_reason`` (set when
#: ``status`` is ``None``) ∈ ``no_value``/``partial_inputs``/``evaluation_error``.
TopologyItem = tuple[str, int, bool | None, str | None]
#: Telemetry key: ``(object_id, kind)``; ``kind`` ∈ ``v``, ``p_from``/``p_to``/
#: ``q_from``/``q_to``, ``p_gen``/``p_load``/``q_gen``/``q_load`` (optionally
#: ``"<kind>:<n>"`` for injection components).
TelemetryKey = tuple[int, str]
#: Telemetry value: ``(value | None, n_inputs, source_guid, quality)``.
TelemetryValue = tuple[float | None, int, str, int]
#: Flow variance factor key: ``(branch_id, flow_kind)``.
FlowKey = tuple[int, str]


@dataclass
class DerivedInputs:
    """Numeric plans derived outside the kernel (see the module docstring).

    Each field is a ready numeric plan for the corresponding step (or ``None`` if the
    step is disabled). The ``run()`` steps apply them via the contract kernels:

    * ``topology_resolved`` — status plan, a list of :data:`TopologyItem`;
    * ``telemetry_resolved`` (``{TelemetryKey: TelemetryValue}``) /
      ``telemetry_arg_keys`` (application order, a list of :data:`TelemetryKey`) /
      ``telemetry_total_args`` (number of source values, statistics only) —
      measurement z-vector;
    * ``materialize_obs`` — observed node injections ``{quantity → {node_id → value}}``,
      ``quantity`` ∈ ``load_p``/``load_q``/``gen_p``/``gen_q``;
    * ``voltage_nominal`` — ``{node_id → vn}`` (off-by-default step);
    * ``v_sigma2_scale`` — ``{node_id → factor}`` of the per-node σ² scale for nodal
      V measurements (factor<1 increases trust in the measurement); ``None`` = no scaling;
    * ``flow_sigma2_scale`` — ``{(branch_id, flow_kind) → factor}`` of the same scale
      for branch flow measurements (flow_kind ∈ ``p_from``/``p_to``/``q_from``/``q_to``).

    The vocabulary, dtypes and meaning of every field are specified in
    :mod:`gridstate.contract.derived_tables`. Tap changer positions are applied
    through the ``tap_steps`` input table (the tap choice was made by the data
    producer), not through a field of this container.
    """

    topology_resolved: list[TopologyItem] | None = None
    telemetry_resolved: dict[TelemetryKey, TelemetryValue] | None = None
    telemetry_arg_keys: list[TelemetryKey] | None = None
    telemetry_total_args: int = 0
    materialize_obs: dict[str, dict[int, float]] | None = None
    voltage_nominal: dict[int, float] | None = None
    v_sigma2_scale: dict[int, float] | None = None
    flow_sigma2_scale: dict[FlowKey, float] | None = None
