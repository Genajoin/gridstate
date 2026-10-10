"""Plain-int code aliases stay equal to the ``gridstate.constants`` enums."""

from __future__ import annotations

from gridstate import z_vector
from gridstate.constants import MeasurementObjectType, MeasurementType
from gridstate.telemetry import _specs


def test_z_vector_kind_and_object_codes() -> None:
    for member in MeasurementType:
        alias = getattr(z_vector, f"KIND_{member.name}")
        assert type(alias) is int
        assert alias == int(member)
    for name in ("NODE", "BRANCH", "GENERATOR"):
        alias = getattr(z_vector, f"OBJ_{name}")
        assert type(alias) is int
        assert alias == int(MeasurementObjectType[name])


def test_telemetry_kind_tables() -> None:
    assert _specs._INJ_MT == {"P": 4, "Q": 5}
    assert _specs._KIND_MAP == {
        "U": (0, 2, -1, 1),
        "PBEG": (1, 0, 0, 1),
        "PEND": (1, 0, 1, 1),
        "QBEG": (1, 1, 0, 1),
        "QEND": (1, 1, 1, 1),
    }
    for codes in _specs._KIND_MAP.values():
        assert all(type(c) is int for c in codes)
