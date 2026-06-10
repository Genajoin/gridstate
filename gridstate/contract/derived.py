"""``DerivedInputs`` — числовые планы входов SE, предвычисленные вне ядра ``run()``.

Контейнер результата обработки источника данных: топология/РПН/телеметрия/
материализация/Vnom уже сведены к числам (через внешний производитель данных) и
сериализуются в граничный файл (``.npz``, см. :mod:`gridstate.contract.serialize`).
Ядро SE применяет эти планы контрактными ядрами (``apply_*_resolved``), не зная,
из какого формата источника они получены.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DerivedInputs:
    """Числовые планы, выведенные вне ядра (см. модульный docstring).

    Каждое поле — готовый числовой план соответствующего шага (или ``None``, если
    шаг отключён). Шаги ``run()`` применяют их контрактными ядрами:

    * ``topology_resolved`` — план статусов ON_LINE;
    * ``telemetry_resolved`` / ``telemetry_arg_keys`` / ``telemetry_total_args`` —
      z-вектор измерений;
    * ``materialize_obs`` — наблюдаемый узловой режим;
    * ``voltage_nominal`` — ``{node_id → vn}`` (off-by-default шаг).

    Применение РПН идёт через входную таблицу ``tap_steps`` (выбор отпайки сделал
    производитель данных), а не через поле этого контейнера.
    """

    topology_resolved: list | None = None
    telemetry_resolved: dict | None = None
    telemetry_arg_keys: list | None = None
    telemetry_total_args: int = 0
    materialize_obs: dict | None = None
    voltage_nominal: dict | None = None
