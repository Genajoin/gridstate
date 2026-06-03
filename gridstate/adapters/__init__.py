"""Адаптеры внешних форматов сети → контракт gridstate (``SEInput``).

Адаптеры — **не** часть рантайма SE: они конвертируют чужой формат описания
сети (например, ``pandapower``-``net``) в контрактные структурированные массивы
:data:`gridstate.contract.tables.SE_INPUT` и оборачивают их в ``SEInput`` через
:meth:`gridstate.working.Working.from_arrays`. Сам пакет ``gridstate`` остаётся
numpy/scipy-only; внешняя библиотека (pandapower) импортируется **лениво** внутри
адаптера и нужна лишь тому, кто строит фикстуры (dev-extra ``[test-models]``).
"""

from __future__ import annotations

from gridstate.adapters.from_pandapower import from_pandapower, measurement_array


__all__ = ["from_pandapower", "measurement_array"]
