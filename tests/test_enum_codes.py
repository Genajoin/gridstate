"""Plain-int code tables stay equal to the ``gridstate.constants`` enums."""

from __future__ import annotations

import pytest

from gridstate import z_vector
from gridstate.constants import MeasurementObjectType, MeasurementQuality, MeasurementType
from gridstate.telemetry import kinds
from gridstate.telemetry.quality import (
    aggregate_qualities,
    inverse_classifier,
    passthrough_classifier,
    strict_classifier,
)


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
    assert kinds.INJ_MT == {"P": 4, "Q": 5}
    assert kinds.NODE_INJ_MAP == {"PG": ("P", 1), "PN": ("P", -1), "QG": ("Q", 1), "QN": ("Q", -1)}
    assert kinds.KIND_MAP == {
        "U": (0, 2, -1, 1),
        "PBEG": (1, 0, 0, 1),
        "PEND": (1, 0, 1, 1),
        "QBEG": (1, 1, 0, 1),
        "QEND": (1, 1, 1, 1),
    }
    for codes in kinds.KIND_MAP.values():
        assert all(type(c) is int for c in codes)
    assert all(type(c) is int for c in kinds.INJ_MT.values())


@pytest.mark.parametrize("table", ["KIND_MAP", "NODE_INJ_MAP", "INJ_MT"])
def test_telemetry_kind_tables_are_read_only(table: str) -> None:
    with pytest.raises(TypeError):
        getattr(kinds, table)["X"] = None


def test_quality_classifiers_return_plain_int_codes() -> None:
    codes = [
        strict_classifier("0x2"),
        strict_classifier("0x1"),
        strict_classifier(""),
        inverse_classifier("0x0"),
        passthrough_classifier("0x0"),
        aggregate_qualities([]),
    ]
    assert all(type(c) is int for c in codes)
    assert codes[:3] == [
        MeasurementQuality.GOOD,
        MeasurementQuality.QUESTIONABLE,
        MeasurementQuality.BAD,
    ]
