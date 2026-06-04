"""Сериализация ``SEInput`` (контракт + ``DerivedInputs``) в ``.npz`` — граница входа.

**Цель.** Единственная граница входа gridstate — **файл данных** (``.npz``),
который готовит внешний инструмент-источник и который gridstate читает **без
внешних зависимостей и без XML**.

Сериализуется ровно то, что нужно ядру для прогона ``run(SEInput)``:

* контрактные таблицы ``SE_INPUT`` — ``nodes`` / ``branches`` / ``measurements`` /
  ``generators`` + доменные ``tap_steps`` / ``load_characteristics`` / ``shunts``
  (структурированные numpy-массивы);
* :class:`~gridstate.contract.derived.DerivedInputs` — 5 числовых планов (топология / РПН /
  телеметрия / материализация / Vnom), результат обработки источника. ``snapshot``
  НЕ сохраняется: ядро его не читает (только косметический счётчик ``unique_guids``).

Загрузчик восстанавливает рабочий слой через :meth:`gridstate.working.Working.from_arrays`
(конструктор из массивов) → ``run()`` исполняется без внешних зависимостей.

**Формат планов (v0, провизорный).** ``DerivedInputs`` несёт dict с tuple-ключами
(``telemetry_resolved``), поэтому планы кодируются ``pickle`` в object-массиве внутри
``.npz``. Это внутренний Python-формат; стабильный кросс-тул формат (JSON-схема
планов) — на будущее. Контрактные ТАБЛИЦЫ хранятся как обычные npz-массивы и читаются
любым инструментом.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from gridstate.contract.runtime import SEInput


_DERIVED_KEY = "__derived_pickle__"
_META_KEY = "__contract_version__"
_CONTRACT_TABLES = ("nodes", "branches", "measurements", "generators")
# Доменные числовые input-таблицы — first-class, наравне с основными
# контрактными: пишутся/читаются под собственным именем. Опциональны
# (источник может их не нести): отсутствующая → пустая коллекция в
# Working.from_arrays.
_DOMAIN_TABLES = ("tap_steps", "load_characteristics", "shunts")


def _derived_to_blob(derived: Any) -> dict | None:
    """``DerivedInputs`` → сериализуемый dict (без ``snapshot`` — ядру не нужен)."""
    if derived is None:
        return None
    return {
        "topology_resolved": derived.topology_resolved,
        "rpn_resolved": derived.rpn_resolved,
        "telemetry_resolved": derived.telemetry_resolved,
        "telemetry_arg_keys": derived.telemetry_arg_keys,
        "telemetry_total_args": derived.telemetry_total_args,
        "materialize_obs": derived.materialize_obs,
        "voltage_nominal": derived.voltage_nominal,
        "snapshot_size": len(derived.snapshot) if derived.snapshot else 0,
    }


def _blob_to_derived(blob: dict | None) -> Any:
    """Сериализованный dict → ``DerivedInputs`` (``snapshot`` восстанавливается пустым).

    ``snapshot`` ядро не читает (см. модульный docstring) — пустой dict даёт лишь
    ``unique_guids=0`` в репорте шага телеметрии (косметика), прогон идентичен.
    """
    if blob is None:
        return None
    from gridstate.contract.derived import DerivedInputs

    return DerivedInputs(
        snapshot={},
        topology_resolved=blob["topology_resolved"],
        rpn_resolved=blob["rpn_resolved"],
        telemetry_resolved=blob["telemetry_resolved"],
        telemetry_arg_keys=blob["telemetry_arg_keys"],
        telemetry_total_args=blob["telemetry_total_args"],
        materialize_obs=blob["materialize_obs"],
        voltage_nominal=blob["voltage_nominal"],
    )


def save_se_input(se_input: SEInput, path: str | Path) -> Path:
    """Записать ``SEInput`` (контрактные таблицы + ``DerivedInputs``) в ``.npz``.

    Args:
        se_input: вход SE. ``se_input.model`` — носитель контрактных таблиц
            (``Working`` или любой объект с коллекциями
            ``.nodes/.branches/.measurements/.generators`` + опц. доменными
            ``.tap_steps/.load_characteristics/.shunts``, отдающими ``to_numpy()``).
            ``se_input.derived`` — числовые планы (или ``None``).
        path: путь к выходному ``.npz`` (расширение добавит numpy при отсутствии).

    Returns:
        Фактический путь записанного файла.
    """
    model = se_input.model
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTRACT_TABLES:
        coll = getattr(model, name)
        arrays[name] = np.asarray(coll.to_numpy())

    # Доменные числовые таблицы — first-class, под собственным именем
    # (опционально: источник может их не нести / нести пустыми).
    for name in _DOMAIN_TABLES:
        coll = getattr(model, name, None)
        if coll is not None and hasattr(coll, "to_numpy"):
            arr = np.asarray(coll.to_numpy())
            if len(arr) > 0:
                arrays[name] = arr

    blob = _derived_to_blob(se_input.derived)
    arrays[_DERIVED_KEY] = np.frombuffer(pickle.dumps(blob), dtype=np.uint8)
    arrays[_META_KEY] = np.asarray(str(se_input.contract_version))

    out = Path(path)
    # mypy: **arrays статически коллидирует с keyword-only allow_pickle: bool в
    # savez; ключи arrays — только имена data-массивов, allow_pickle среди них нет.
    np.savez(out, **arrays)  # type: ignore[arg-type]
    # numpy дописывает .npz, если расширения нет — вернём фактическое имя.
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def load_se_input_npz(path: str | Path) -> SEInput:
    """Прочитать ``.npz`` (см. :func:`save_se_input`) в ``SEInput`` — БЕЗ внешних зависимостей и XML.

    Рабочий слой собирается через :meth:`gridstate.working.Working.from_arrays`.
    Возвращаемый ``SEInput`` готов к ``run(se_input)``: ``derived`` —
    восстановленные числовые планы.
    """
    from gridstate.contract.runtime import SEInput
    from gridstate.working import Working

    with np.load(path, allow_pickle=False) as npz:
        files = set(npz.files)
        domain = {
            name: np.asarray(npz[name]) for name in _DOMAIN_TABLES if name in files
        }
        working = Working.from_arrays(
            nodes=np.asarray(npz["nodes"]),
            branches=np.asarray(npz["branches"]),
            measurements=np.asarray(npz["measurements"]),
            generators=np.asarray(npz["generators"]),
            **domain,
        )
        blob = pickle.loads(bytes(npz[_DERIVED_KEY])) if _DERIVED_KEY in files else None
        contract_version = str(npz[_META_KEY]) if _META_KEY in files else None

    derived = _blob_to_derived(blob)
    kwargs: dict[str, Any] = {"model": working, "derived": derived}
    if contract_version is not None:
        kwargs["contract_version"] = contract_version
    return SEInput(**kwargs)
