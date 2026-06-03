"""Доменные перечисления gridstate (контракт владеет ими — target-arch §5.2).

gridstate — самостоятельный модуль SE; домен-перечисления (типы узлов/ветвей/
измерений) фиксируются **у нас**, а не тянутся из ``power-system-core``. Это
снимает рантайм-импорт PSC из ядра (Фаза 5): модули телеметрии/препроцессинга
ссылаются на эти константы, а не на ``power_system.utils.constants``.

Значения IntEnum СОВПАДАЮТ с ``power_system.utils.constants`` **1:1** — они
участвуют в целочисленных сравнениях z-вектора/фильтров/контракта, поэтому
расхождение сломало бы бит-в-бит. Страж дрейфа (пока PSC в окружении) —
``tests/test_constants_match_psc.py``.

(``topology.py`` исторически инлайнит те же литералы с комментарием
``# NodeType.SLACK``; это эквивалентно. Унификация на этот модуль — опц. чистка Ф7.)
"""

from __future__ import annotations

from enum import IntEnum


class NodeType(IntEnum):
    """Тип узла (``node_type``-колонка контракта)."""

    PQ = 0
    PV = 1
    SLACK = 2


class BranchType(IntEnum):
    """Тип ветви (``branch_type``-колонка контракта)."""

    LINE = 0
    TRANSFORMER = 1
    REACTOR = 2


class MeasurementType(IntEnum):
    """Тип измерения (``measurement_type``-колонка контракта)."""

    POWER_P = 0
    POWER_Q = 1
    VOLTAGE = 2
    CURRENT = 3
    POWER_INJECTION_P = 4
    POWER_INJECTION_Q = 5


class MeasurementObjectType(IntEnum):
    """Тип объекта измерения (``object_type``-колонка контракта)."""

    NODE = 0
    BRANCH = 1
    GENERATOR = 2
    REACTOR = 3
    SECTION = 4
    LPA_ELEMENT = 5
    LPA_DEVICE = 6
    AVPROC = 7
    VIR = 8
    TRANZIT = 9
    REMONT = 10
    LPA_SUBELEMENT = 11
    NONE = -1
