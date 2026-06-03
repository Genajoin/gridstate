"""Страж границы рабочего слоя (Ф4 target-architecture): INPUT/KEY-колонки контракта
НЕ мутируются пайплайном — мутируется только WORKING/OUTPUT.

Это **инвариант, оправдывающий «узкий clone» Ф4.0** (копировать лишь WORKING+KEY,
INPUT держать shared/read-only) и каждый последующий пер-функциональный перенос
Ф4.1+: если хоть один production-шаг пишет в колонку, помеченную в контракте
``Role.INPUT``/``Role.KEY`` (gridstate/contract/tables.py), — это либо неверная
разметка контракта, либо шаг, который при снятии clone начнёт мутировать Input.
Страж ловит оба случая БЕЗ canon (per-phase, см. feedback_defer_heavy_gate_migration).

Метод: прогнать полный ``run`` (Input read-only → working-копия в ``result.model``),
сверить по ключу ``id`` все INPUT+KEY-колонки 4 коллекций между Input и working-
финалом. Пробел контракта по RAW_TABLES (``reactors.status`` мутирует шаг
ON_LINE-топологии) сюда НЕ попадает: это raw-таблица, не одна из 4 основных
коллекций, и под Ф4.0 она всё ещё deepcopy-ится (решается на Ф4.N).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT, Role
from gridstate.pipeline import PipelineConfig, run

# Переиспользуем синтетику из idempotent-сьюта.
from tests.test_pipeline_idempotent import _make_model_with_reactor


def _assert_input_key_columns_unchanged(input_model, working_model) -> int:
    """Сверить: каждая INPUT/KEY-колонка контракта бит-в-бит между Input и working.

    Сопоставление по ``id`` (measurements в working могут иметь ДОБАВЛЕННЫЕ
    pseudo-строки — их не проверяем; исходные id обязаны сохраниться). Возвращает
    число проверенных (collection, column) пар — для liveness-проверки.
    """
    pairs = [
        (SE_INPUT.nodes, input_model.nodes, working_model.nodes),
        (SE_INPUT.branches, input_model.branches, working_model.branches),
        (SE_INPUT.measurements, input_model.measurements, working_model.measurements),
        (SE_INPUT.generators, input_model.generators, working_model.generators),
    ]
    checked = 0
    for schema, in_coll, wk_coll in pairs:
        in_arr = in_coll.to_numpy()
        if len(in_arr) == 0:
            continue
        wk_arr = wk_coll.to_numpy()
        present = set(in_arr.dtype.names or ()) & set(wk_arr.dtype.names or ())
        cols = [c for c in schema.column_names(Role.KEY, Role.INPUT) if c in present]
        wk_by_id = {int(row["id"]): row for row in wk_arr}
        for row in in_arr:
            rid = int(row["id"])
            assert rid in wk_by_id, f"{schema.name} id={rid} исчез в working-копии"
            wrow = wk_by_id[rid]
            for c in cols:
                assert np.array_equal(row[c], wrow[c]), (
                    f"{schema.name}.{c} (роль INPUT/KEY) изменилась на id={rid}: "
                    f"{row[c]!r} → {wrow[c]!r} — либо неверная разметка контракта, "
                    f"либо шаг мутирует Input"
                )
            checked += len(cols)
    return checked


# ---------------------------------------------------------------------------
# Синтетика (всегда): покрывает non-XML шаги пайплайна
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_input_key_columns_immutable_synthetic(algorithm):
    """Полный run на синтетике: INPUT/KEY-колонки 4 коллекций не тронуты."""
    m = _make_model_with_reactor()
    n_meas_in = len(m.measurements.to_numpy())

    r = run(m, config=PipelineConfig(algorithm=algorithm))

    checked = _assert_input_key_columns_unchanged(m, r.model)
    assert checked > 0, "страж ничего не проверил — список INPUT/KEY-колонок пуст?"
    # Liveness: пайплайн реально что-то сделал в WORKING-слое (добавил pseudo-меры
    # и/или записал shunt) — иначе страж проходил бы вакуумно.
    assert len(r.model.measurements.to_numpy()) >= n_meas_in


def test_working_columns_do_change_synthetic():
    """Контроль: WORKING-колонки (shunt_b от реактора) действительно меняются —
    подтверждает, что страж выше не вакуумен (Input/working вообще различимы)."""
    m = _make_model_with_reactor()
    r = run(m, config=PipelineConfig(algorithm="wls"))
    # реактор на узле 1 → working shunt_b != 0, Input shunt_b == 0.
    assert float(m.nodes.get_by_id(1).shunt_b) == pytest.approx(0.0)
    assert float(r.model.nodes.get_by_id(1).shunt_b) != pytest.approx(0.0)
