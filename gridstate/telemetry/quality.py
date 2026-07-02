"""Классификация кодов качества телеметрии ``tm_code`` → ``MeasurementQuality``.

Каждое значение телеметрии приходит с полем ``tm_code`` — hex-строкой 32-битной
маски качества. Этот модуль конвертирует hex-строку в целочисленный класс
качества (``GOOD/QUESTIONABLE/BAD``), который дальше используется при применении
телеметрии для:

* отбрасывания недостоверных measurements (``status=False``);
* понижения веса сомнительных (увеличение ``σ²``).

## Семантика qCode

Полной публичной спецификации значений всех битов маски нет. Из наблюдений
зафиксировано поведение младших двух бит:

* ``qCode == 0`` — sentinel «нет данных»;
* ``(qCode & 3) == 2`` — достоверное значение (любые старшие биты допустимы,
  в том числе бит 16 = «ручной ввод с блокировкой»);
* иначе — частично достоверное (``{0, 1, 3}`` в младших битах при ``qCode != 0``).

| Условие | Класс |
|---|---|
| ``qCode == 0`` | BAD (нет данных) |
| ``(qCode & 3) == 2`` | GOOD (включая бит 16 ручного ввода) |
| иначе | QUESTIONABLE |

## Default-классификатор

По умолчанию применяется :func:`strict_classifier`. Это безопасно: на свежих
снимках доля BAD по строгой логике — 0.00–0.06 %, доля QUESTIONABLE —
0.09–0.20 %, итоговый функционал J SE меняется в пределах −0.06 %.

Если нужно полностью отключить фильтрацию по качеству (например, для диагностики
или для снимков, состоящих из устаревших точек с ``qCode == 0``), используйте
:func:`passthrough_classifier`.

Перед сменой классификатора снимите histogram через :func:`tm_code_histogram` —
он покажет, сколько measurements попадёт в каждый класс на конкретном snapshot.
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

# Имя класса по числу — для tm_code_histogram.
_NAMES = {
    QUALITY_GOOD: "GOOD",
    QUALITY_QUESTIONABLE: "QUESTIONABLE",
    QUALITY_BAD: "BAD",
}


def _parse_code(code_hex: str | None) -> int | None:
    """Hex/dec строка → int, либо ``None`` если строка пустая/невалидная."""
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
    """Default-классификатор. Строгая логика по младшим битам маски.

    * ``qCode == 0`` → BAD (sentinel «нет данных»);
    * ``(qCode & 3) == 2`` → GOOD (любые старшие биты допустимы,
      в т.ч. бит 16 «ручной ввод с блокировкой»);
    * иначе → QUESTIONABLE (``(qCode & 3) ∈ {0, 1, 3}`` при ``qCode != 0``).
    """
    code = _parse_code(code_hex)
    if code is None or code == 0:
        return QUALITY_BAD
    if (code & 0x3) == 2:
        return QUALITY_GOOD
    return QUALITY_QUESTIONABLE


def passthrough_classifier(_code_hex: str) -> int:
    """Отключает фильтрацию по качеству: всё → GOOD.

    Полезно для диагностики, для отключения tm_code-фильтра в тестах
    либо для снимков, состоящих преимущественно из устаревших точек с
    ``qCode == 0``, где ``strict_classifier`` отбросил бы большую часть данных.
    """
    return QUALITY_GOOD


def inverse_classifier(code_hex: str) -> int:
    """Эмпирическая интерпретация: ``qCode == 0`` → GOOD, иначе → QUESTIONABLE.

    Подходит только для случаев, когда подавляющее большинство точек snapshot-а
    имеют ``qCode == 0`` и наблюдается, что SE на этих точках сходится корректно
    (т.е. de-facto это «достоверные» данные с ненормализованным кодом).
    Не общеприменимо.
    """
    code = _parse_code(code_hex)
    if code is None:
        return QUALITY_BAD
    if code == 0:
        return QUALITY_GOOD
    return QUALITY_QUESTIONABLE


def aggregate_qualities(qualities: list[int]) -> int:
    """Worst-case по списку классов: любой BAD → BAD; любой QUESTIONABLE → QUESTIONABLE.

    Пустой список → GOOD (нечего ухудшать).
    """
    if not qualities:
        return QUALITY_GOOD
    return max(qualities)


def tm_code_histogram(
    snapshot: dict[str, object],
    classifier: Callable[[str], int] = strict_classifier,
) -> dict[str, int]:
    """Подсчитать распределение классов в snapshot для данного classifier.

    Args:
        snapshot: ``dict[guid → объект]``, у каждого значения берётся атрибут
            ``tm_code`` (строка кода качества).
        classifier: функция ``str → int`` (один из заранее определённых или
            пользовательский). По умолчанию — ``strict_classifier``.

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
