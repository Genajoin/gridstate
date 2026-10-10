"""Classification of telemetry quality codes ``tm_code`` → ``MeasurementQuality``.

Every telemetry value arrives with a ``tm_code`` field, a hex string of a 32-bit
quality mask. This module converts the hex string into an integer quality
class (``GOOD/QUESTIONABLE/BAD``), which is then used when applying
telemetry to:

* drop unreliable measurements (``status=False``);
* down-weight questionable ones (increase ``σ²``).

## qCode semantics

There is no complete public specification of all mask bits. The behaviour of the
two low bits has been recorded from observations:

* ``qCode == 0`` is the "no data" sentinel;
* ``(qCode & 3) == 2`` is a reliable value (any high bits are allowed,
  including bit 16 = "manual entry with lock");
* otherwise the value is partially reliable (``{0, 1, 3}`` in the low bits with ``qCode != 0``).

| Condition | Class |
|---|---|
| ``qCode == 0`` | BAD (no data) |
| ``(qCode & 3) == 2`` | GOOD (including bit 16, manual entry) |
| otherwise | QUESTIONABLE |

## Default classifier

:func:`strict_classifier` is applied by default. This is safe: on fresh
snapshots the BAD share under the strict logic is 0.00–0.06 %, the QUESTIONABLE share is
0.09–0.20 %, and the final SE objective J changes within −0.06 %.

If quality filtering has to be disabled completely (for example for diagnostics
or for snapshots made of stale points with ``qCode == 0``), use
:func:`passthrough_classifier`.

Before switching the classifier, take a histogram via :func:`tm_code_histogram`:
it shows how many measurements fall into each class on a specific snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

from gridstate.constants import MeasurementQuality


__all__ = [
    "QUALITY_BAD",
    "QUALITY_GOOD",
    "QUALITY_QUESTIONABLE",
    "aggregate_qualities",
    "inverse_classifier",
    "passthrough_classifier",
    "strict_classifier",
    "tm_code_histogram",
]


# Canonical values live in gridstate.constants.MeasurementQuality; these
# module-level ints are kept as backward-compatible aliases.
QUALITY_GOOD = int(MeasurementQuality.GOOD)
QUALITY_QUESTIONABLE = int(MeasurementQuality.QUESTIONABLE)
QUALITY_BAD = int(MeasurementQuality.BAD)

# Class name by number, for tm_code_histogram.
_NAMES = {
    QUALITY_GOOD: "GOOD",
    QUALITY_QUESTIONABLE: "QUESTIONABLE",
    QUALITY_BAD: "BAD",
}


def _parse_code(code_hex: str | None) -> int | None:
    """Hex/dec string → int, or ``None`` if the string is empty or invalid."""
    if not code_hex:
        return None
    s = code_hex.strip()
    if not s:
        return None
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s)
    except ValueError:
        return None


def strict_classifier(code_hex: str) -> int:
    """Default classifier. Strict logic on the low bits of the mask.

    * ``qCode == 0`` → BAD ("no data" sentinel);
    * ``(qCode & 3) == 2`` → GOOD (any high bits are allowed,
      including bit 16 "manual entry with lock");
    * otherwise → QUESTIONABLE (``(qCode & 3) ∈ {0, 1, 3}`` with ``qCode != 0``).
    """
    code = _parse_code(code_hex)
    if code is None or code == 0:
        return QUALITY_BAD
    if (code & 0x3) == 2:
        return QUALITY_GOOD
    return QUALITY_QUESTIONABLE


def passthrough_classifier(_code_hex: str) -> int:
    """Disable quality filtering: everything → GOOD.

    Useful for diagnostics, for turning off the tm_code filter in tests,
    or for snapshots made mostly of stale points with
    ``qCode == 0``, where ``strict_classifier`` would drop most of the data.
    """
    return QUALITY_GOOD


def inverse_classifier(code_hex: str) -> int:
    """Empirical interpretation: ``qCode == 0`` → GOOD, otherwise → QUESTIONABLE.

    Suitable only when the vast majority of snapshot points
    have ``qCode == 0`` and it is observed that SE converges correctly on these points
    (i.e. de facto this is reliable data with a non-normalized code).
    Not generally applicable.
    """
    code = _parse_code(code_hex)
    if code is None:
        return QUALITY_BAD
    if code == 0:
        return QUALITY_GOOD
    return QUALITY_QUESTIONABLE


def aggregate_qualities(qualities: list[int]) -> int:
    """Worst case over a list of classes: any BAD → BAD; any QUESTIONABLE → QUESTIONABLE.

    An empty list → GOOD (nothing to worsen).
    """
    if not qualities:
        return QUALITY_GOOD
    return max(qualities)


def tm_code_histogram(
    snapshot: dict[str, object],
    classifier: Callable[[str], int] = strict_classifier,
) -> dict[str, int]:
    """Count the class distribution in a snapshot for the given classifier.

    Args:
        snapshot: ``dict[guid → object]``; the ``tm_code`` attribute (a quality
            code string) is taken from each value.
        classifier: a ``str → int`` function (one of the predefined ones or
            a custom one). Defaults to ``strict_classifier``.

    Returns:
        ``{"GOOD": N, "QUESTIONABLE": N, "BAD": N, "total": N}``.
    """
    counts = {"GOOD": 0, "QUESTIONABLE": 0, "BAD": 0}
    total = 0
    for tv in snapshot.values():
        total += 1
        q = classifier(getattr(tv, "tm_code", "") or "")
        counts[_NAMES.get(q, "BAD")] += 1
    counts["total"] = total
    return counts
