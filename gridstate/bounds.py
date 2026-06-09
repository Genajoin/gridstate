"""Единая семантика «незаданных» границ box-полей узла.

Контрактная конвенция (``NODE_DTYPE``: ``load_p_min/max``,
``generation_q_min/max`` и т.д.): пара границ считается **не заданной**,
если оба значения нулевые (numpy-default незаполненного поля) либо
|значение| ≥ сентинела (±9999 — «нет данных» в исходных форматах).

До этого модуля каждая подсистема трактовала границы по-своему:
``ipm_setup._bound_pair_or_none`` понимал нули и сентинелы, а клип
WLS-разноса (``post_processing._clip``) — нет, из-за чего незаполненная
пара (0, 0) **зануляла** оценку, а сентинельная ±9999 давала мусорные
суммы в агрегации генераторов. Здесь — одно правило для всех.
"""

from __future__ import annotations


__all__ = ["SENTINEL_ABS", "is_sentinel", "resolve_bounds"]

#: |значение| ≥ этого — «не задано» (контрактная конвенция ±9999).
SENTINEL_ABS = 9000.0


def is_sentinel(value: float, *, sentinel_abs: float = SENTINEL_ABS) -> bool:
    """Является ли значение сентинелом «нет данных»."""
    return abs(float(value)) >= sentinel_abs


def resolve_bounds(
    lo_raw: float,
    hi_raw: float,
    *,
    sentinel_abs: float = SENTINEL_ABS,
) -> tuple[float, float]:
    """Пара ``(lo, hi)`` с заменой незаданных сторон на ±inf.

    Правила:

    * обе стороны ≈0 → ``(-inf, +inf)`` — пара не заполнялась
      (numpy-default); реальный диапазон «ровно [0, 0]» у активного
      узла не встречается — такой узел не несёт ``exist_*``;
    * ОБЕ стороны |..| ≥ ``sentinel_abs`` → пара не задана →
      ``(-inf, +inf)``. Полусентинельная пара (одна сторона большая)
      сохраняется **как есть**: реальные BUS-эквиваленты несут границы
      порядка ±десятков ГВт (42 500 МВт > сентинела), и резать их
      нельзя — большая сторона неотличима от данных;
    * ``lo > hi`` → ``(-inf, +inf)`` — границы некорректны, не
      ограничиваем (раньше такие пары либо зануляли оценку, либо
      молча пропускались).

    Потребители: клип WLS-разноса (``±inf`` = «не клиповать»), box-vars
    IPM (``±inf`` = «подставить широкий дефолт»).
    """
    lo = float(lo_raw)
    hi = float(hi_raw)
    if abs(lo) < 1e-9 and abs(hi) < 1e-9:
        return (float("-inf"), float("inf"))
    if abs(lo) >= sentinel_abs and abs(hi) >= sentinel_abs:
        return (float("-inf"), float("inf"))
    if lo > hi:
        return (float("-inf"), float("inf"))
    return (lo, hi)
