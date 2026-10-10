"""``DerivedInputs`` — numeric plans of the SE inputs, precomputed outside the ``run()`` kernel.

A container for the result of processing a data source: topology/tap changers/telemetry/
materialization/Vnom are already reduced to numbers (by an external data producer) and
serialized into a boundary file (``.npz``, see :mod:`gridstate.contract.serialize`).
The SE kernel applies these plans through the contract kernels (``apply_*_resolved``) without knowing
which source format they came from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DerivedInputs:
    """Numeric plans derived outside the kernel (see the module docstring).

    Each field is a ready numeric plan for the corresponding step (or ``None`` if the
    step is disabled). The ``run()`` steps apply them via the contract kernels:

    * ``topology_resolved`` — plan of ON_LINE statuses;
    * ``telemetry_resolved`` / ``telemetry_arg_keys`` / ``telemetry_total_args`` —
      measurement z-vector;
    * ``materialize_obs`` — observable nodal state;
    * ``voltage_nominal`` — ``{node_id → vn}`` (off-by-default step);
    * ``v_sigma2_scale`` — ``{node_id → factor}`` of the per-node σ² scale for nodal
      V measurements (factor<1 increases trust in the measurement); ``None`` = no scaling;
    * ``flow_sigma2_scale`` — ``{(branch_id, kind) → factor}`` of the same scale
      for branch flow measurements (kind ∈ PBEG/PEND/QBEG/QEND).

    Tap changer positions are applied through the ``tap_steps`` input table (the tap choice was made by
    the data producer), not through a field of this container.
    """

    topology_resolved: list | None = None
    telemetry_resolved: dict | None = None
    telemetry_arg_keys: list | None = None
    telemetry_total_args: int = 0
    materialize_obs: dict | None = None
    voltage_nominal: dict | None = None
    v_sigma2_scale: dict | None = None
    flow_sigma2_scale: dict | None = None
