"""Подготовка модели к WLS: псевдо-измерения и properties входного формата.

Топологическая чистка (`disable_orphan_branches`, `disable_disconnected_components`,
`disable_isolated_nodes`, `refine_slack_to_one`, `refine_node_types_from_generators`)
живёт в ``gridstate.topology``. Используется напрямую::

    from gridstate.topology import (
        disable_orphan_branches,
        disable_disconnected_components,
        disable_isolated_nodes,
        refine_slack_to_one,
        refine_node_types_from_generators,
    )

SE-специфичные шаги (псевдо-измерения, чтение properties входного формата) — здесь.
"""

from gridstate.preprocessing.mirror_voltage import mirror_voltage_through_unit_tap_links
from gridstate.preprocessing.node_props import extract_node_load_props_from_model
from gridstate.preprocessing.pseudo_measurements import add_pseudo_measurements


__all__ = [
    "add_pseudo_measurements",
    "extract_node_load_props_from_model",
    "mirror_voltage_through_unit_tap_links",
]
