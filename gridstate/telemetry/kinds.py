"""Telemetry kind maps: reading kind → measurement coordinates in ``SE_INPUT.measurements``.

A telemetry plan (``DerivedInputs.telemetry_resolved``) keys every reading by a
string kind. These read-only tables give the numeric meaning of each kind (codes
from the :mod:`gridstate.constants` enums); the telemetry core
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


__all__ = ["INJ_MT", "KIND_MAP", "NODE_INJ_MAP"]


_NODE = int(MeasurementObjectType.NODE)
_BRANCH = int(MeasurementObjectType.BRANCH)
_P = int(MeasurementType.POWER_P)
_Q = int(MeasurementType.POWER_Q)
_V = int(MeasurementType.VOLTAGE)


#: kind → ``(object_type, measurement_type, branch_side, sign)``.
#:
#: * ``U`` — node voltage: ``object_type`` NODE, ``measurement_type`` VOLTAGE,
#:   ``branch_side`` -1 (none);
#: * ``PBEG`` / ``PEND`` / ``QBEG`` / ``QEND`` — branch P / Q flow at the from
#:   (``branch_side`` 0) or to (``branch_side`` 1) end;
#: * ``sign`` — multiplier applied to the value (+1 for every kind).
KIND_MAP: Final[Mapping[str, tuple[int, int, int, int]]] = MappingProxyType(
    {
        "U": (_NODE, _V, -1, +1),
        "PBEG": (_BRANCH, _P, 0, +1),
        "PEND": (_BRANCH, _P, +1, +1),
        "QBEG": (_BRANCH, _Q, 0, +1),
        "QEND": (_BRANCH, _Q, +1, +1),
    }
)

#: kind → ``(part, sign)`` of a node injection: ``part`` is ``"P"`` or ``"Q"``
#: (a key of :data:`INJ_MT`); ``sign`` is the multiplier of the reading in the
#: net injection: +1 for generation (``PG`` / ``QG``), -1 for load (``PN`` / ``QN``).
NODE_INJ_MAP: Final[Mapping[str, tuple[str, int]]] = MappingProxyType(
    {
        "PG": ("P", +1),
        "PN": ("P", -1),
        "QG": ("Q", +1),
        "QN": ("Q", -1),
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
