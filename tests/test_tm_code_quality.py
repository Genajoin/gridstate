"""Tests for gridstate.telemetry.quality."""

from __future__ import annotations

from dataclasses import dataclass

from gridstate.constants import MeasurementQuality
from gridstate.telemetry import (
    aggregate_qualities,
    inverse_classifier,
    passthrough_classifier,
    strict_classifier,
    tm_code_histogram,
)


class TestPassthroughClassifier:
    """passthrough_classifier - the "disable the filter" option: everything -> GOOD."""

    def test_returns_good_for_zero(self) -> None:
        assert passthrough_classifier("0x0") == MeasurementQuality.GOOD

    def test_returns_good_for_nonzero(self) -> None:
        assert passthrough_classifier("0x70000002") == MeasurementQuality.GOOD

    def test_returns_good_for_empty(self) -> None:
        assert passthrough_classifier("") == MeasurementQuality.GOOD


class TestStrictClassifier:
    def test_zero_is_bad(self) -> None:
        assert strict_classifier("0x0") == MeasurementQuality.BAD

    def test_lower_bits_10_is_good(self) -> None:
        # (qCode & 3) == 2 -> reliable value
        assert strict_classifier("0x70000002") == MeasurementQuality.GOOD
        assert strict_classifier("0x10000002") == MeasurementQuality.GOOD
        assert strict_classifier("0x80000002") == MeasurementQuality.GOOD

    def test_manual_input_bit_is_still_good(self) -> None:
        # Bit 16 (0x10000) is "manual entry with blocking". It does not override
        # the base check (& 3 == 2): manual + good_bits -> GOOD.
        assert strict_classifier("0x10002") == MeasurementQuality.GOOD

    def test_lower_bits_other_is_questionable(self) -> None:
        # (qCode & 3) ∈ {0, 1, 3} (but qCode != 0) -> QUESTIONABLE.
        assert strict_classifier("0x1") == MeasurementQuality.QUESTIONABLE  # & 3 == 1
        assert strict_classifier("0x3") == MeasurementQuality.QUESTIONABLE  # & 3 == 3
        assert (
            strict_classifier("0x10000") == MeasurementQuality.QUESTIONABLE
        )  # manual without good_bits
        assert strict_classifier("0x70000000") == MeasurementQuality.QUESTIONABLE  # & 3 == 0

    def test_empty_or_invalid_is_bad(self) -> None:
        assert strict_classifier("") == MeasurementQuality.BAD
        assert strict_classifier("garbage") == MeasurementQuality.BAD


class TestInverseClassifier:
    def test_zero_is_good(self) -> None:
        # Empirical rule for snapshots where qCode==0 predominates.
        assert inverse_classifier("0x0") == MeasurementQuality.GOOD

    def test_nonzero_is_questionable(self) -> None:
        assert inverse_classifier("0x70000002") == MeasurementQuality.QUESTIONABLE
        assert inverse_classifier("0x1") == MeasurementQuality.QUESTIONABLE

    def test_empty_is_bad(self) -> None:
        assert inverse_classifier("") == MeasurementQuality.BAD


class TestAggregateQualities:
    def test_all_good(self) -> None:
        assert (
            aggregate_qualities([MeasurementQuality.GOOD, MeasurementQuality.GOOD])
            == MeasurementQuality.GOOD
        )

    def test_one_questionable(self) -> None:
        # worst case: a single QUESTIONABLE lowers the class to QUESTIONABLE.
        assert (
            aggregate_qualities(
                [MeasurementQuality.GOOD, MeasurementQuality.GOOD, MeasurementQuality.QUESTIONABLE]
            )
            == MeasurementQuality.QUESTIONABLE
        )

    def test_one_bad(self) -> None:
        # Any BAD -> BAD.
        assert (
            aggregate_qualities(
                [MeasurementQuality.GOOD, MeasurementQuality.QUESTIONABLE, MeasurementQuality.BAD]
            )
            == MeasurementQuality.BAD
        )

    def test_empty_list_returns_good(self) -> None:
        # No ARG -> nothing to degrade -> GOOD (default point).
        assert aggregate_qualities([]) == MeasurementQuality.GOOD


@dataclass
class _TM:
    """Minimal stand-in for ``TmValue`` in histogram tests."""

    tm_code: str


class TestTmCodeHistogram:
    def test_strict_on_synthetic_snapshot(self) -> None:
        snapshot = {
            "g1": _TM("0x0"),  # BAD
            "g2": _TM("0x70000002"),  # GOOD
            "g3": _TM("0x10000002"),  # GOOD
            "g4": _TM("0x10001"),  # QUESTIONABLE (& 3 == 1)
            "g5": _TM(""),  # BAD
        }
        result = tm_code_histogram(snapshot, strict_classifier)
        assert result == {
            "GOOD": 2,
            "QUESTIONABLE": 1,
            "BAD": 2,
            "total": 5,
        }

    def test_passthrough_marks_everything_good(self) -> None:
        snapshot = {
            "g1": _TM("0x0"),
            "g2": _TM("0x70000002"),
            "g3": _TM(""),
        }
        result = tm_code_histogram(snapshot, passthrough_classifier)
        assert result["GOOD"] == 3
        assert result["QUESTIONABLE"] == 0
        assert result["BAD"] == 0
        assert result["total"] == 3

    def test_empty_snapshot(self) -> None:
        assert tm_code_histogram({}, strict_classifier) == {
            "GOOD": 0,
            "QUESTIONABLE": 0,
            "BAD": 0,
            "total": 0,
        }

    def test_default_classifier_is_strict(self) -> None:
        # The default histogram classifier is strict_classifier.
        # 0x70000002 (& 3 == 2) → GOOD; 0x0 → BAD.
        snapshot = {"g1": _TM("0x70000002"), "g2": _TM("0x0")}
        result = tm_code_histogram(snapshot)
        assert result["GOOD"] == 1
        assert result["BAD"] == 1
