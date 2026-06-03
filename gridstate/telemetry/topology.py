"""Фасад обратной совместимости: код переехал в concern-модули (Ф4 раскол).

Этот модуль исторически собирал несколько концернов prep-слоя в одном файле. Он
разбит по концернам на отдельные модули ``gridstate/telemetry/{measurements,shunts,
generators,rpn,on_line}.py``. **Перенос чисто механический — поведение не изменено.**

Здесь оставлен re-export для обратной совместимости (включая приватный символ
``_BREAKER_X_SENTINEL_OHM``, который тянут тесты). Новый код импортируйте напрямую
из concern-модулей.
"""
# Модуль — чистый re-export-фасад: импорты намеренно «неиспользуемы» локально
# (пробрасываются наружу, включая приватные символы вне __all__).
# ruff: noqa: F401

from __future__ import annotations

from gridstate.telemetry._specs import RpnSpec
from gridstate.telemetry.generators import (
    aggregate_generators_to_node,
    apply_generator_status_from_node,
)
from gridstate.telemetry.measurements import (
    deactivate_orphan_measurements,
    resolve_merged_measurement_conflicts,
)
from gridstate.telemetry.shunts import (
    _BREAKER_X_SENTINEL_OHM,
    apply_reactors_to_node_shunt,
    normalize_breaker_reactance,
)


__all__ = [
    "RpnSpec",
    "aggregate_generators_to_node",
    "apply_generator_status_from_node",
    "apply_reactors_to_node_shunt",
    "deactivate_orphan_measurements",
    "normalize_breaker_reactance",
    "resolve_merged_measurement_conflicts",
]
