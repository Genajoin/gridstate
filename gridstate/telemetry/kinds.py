"""Telemetry kind maps: reading kind → measurement coordinates in ``SE_INPUT.measurements``.

A telemetry plan (``DerivedInputs.telemetry_resolved``) keys every reading by a
string kind of the vocabulary of :mod:`gridstate.contract.derived_tables` (the
single source of the kind names: ``v``, ``p_from`` / ``p_to`` / ``q_from`` /
``q_to``, ``p_gen`` / ``p_load`` / ``q_gen`` / ``q_load``; an injection
component may carry a component number, ``"p_gen:3"``, see
:func:`~gridstate.contract.derived_tables.kind_base`). These read-only tables
give the numeric meaning of each kind (codes from the :mod:`gridstate.constants`
enums); the telemetry core
(:func:`gridstate.telemetry.apply_telemetry_resolved`) applies them, and a data
producer building telemetry plans can use the same tables instead of copying
them:

* :data:`KIND_MAP` — node voltage and branch flow kinds;
* :data:`NODE_INJ_MAP` — node injection kinds (generation and load parts);
* :data:`INJ_MT` — measurement type of a net node injection.

The binding records of a particular source format (which source values feed a
reading) are not part of gridstate: the core receives the already computed
numeric plans.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from gridstate.constants import MeasurementObjectType, MeasurementType
from gridstate.contract.derived_tables import (
    KIND_P_FROM,
    KIND_P_GEN,
    KIND_P_LOAD,
    KIND_P_TO,
    KIND_Q_FROM,
    KIND_Q_GEN,
    KIND_Q_LOAD,
    KIND_Q_TO,
    KIND_V,
)


__all__ = ["INJ_MT", "KIND_MAP", "NODE_INJ_MAP"]


_NODE = int(MeasurementObjectType.NODE)
_BRANCH = int(MeasurementObjectType.BRANCH)
_P = int(MeasurementType.POWER_P)
_Q = int(MeasurementType.POWER_Q)
_V = int(MeasurementType.VOLTAGE)


#: kind → ``(object_type, measurement_type, branch_side, sign)``.
#:
#: * ``v`` — node voltage: ``object_type`` NODE, ``measurement_type`` VOLTAGE,
#:   ``branch_side`` -1 (none);
#: * ``p_from`` / ``p_to`` / ``q_from`` / ``q_to`` — branch P / Q flow at the from
#:   (``branch_side`` 0) or to (``branch_side`` 1) end;
#: * ``sign`` — multiplier applied to the value (+1 for every kind).
KIND_MAP: Final[Mapping[str, tuple[int, int, int, int]]] = MappingProxyType(
    {
        KIND_V: (_NODE, _V, -1, +1),
        KIND_P_FROM: (_BRANCH, _P, 0, +1),
        KIND_P_TO: (_BRANCH, _P, +1, +1),
        KIND_Q_FROM: (_BRANCH, _Q, 0, +1),
        KIND_Q_TO: (_BRANCH, _Q, +1, +1),
    }
)

#: injection kind (without its component number) → ``(part, sign)`` of a node
#: injection: ``part`` is ``"P"`` or ``"Q"`` (a key of :data:`INJ_MT`); ``sign`` is
#: the multiplier of the reading in the net injection: +1 for generation
#: (``p_gen`` / ``q_gen``), -1 for load (``p_load`` / ``q_load``). The components
#: of a node (all its kinds and component numbers) are summed.
NODE_INJ_MAP: Final[Mapping[str, tuple[str, int]]] = MappingProxyType(
    {
        KIND_P_GEN: ("P", +1),
        KIND_P_LOAD: ("P", -1),
        KIND_Q_GEN: ("Q", +1),
        KIND_Q_LOAD: ("Q", -1),
    }
)

#: Injection part (``"P"`` / ``"Q"``) → ``measurement_type`` code of the net
#: node injection (``POWER_INJECTION_P`` / ``POWER_INJECTION_Q``).
INJ_MT: Final[Mapping[str, int]] = MappingProxyType(
    {
        "P": int(MeasurementType.POWER_INJECTION_P),
        "Q": int(MeasurementType.POWER_INJECTION_Q),
    }
)
