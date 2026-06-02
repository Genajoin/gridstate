"""Подготовка модели к WLS: псевдо-измерения и properties входного формата.

Топологическая чистка (`disable_orphan_branches`, `disable_disconnected_components`,
`disable_isolated_nodes`, `refine_slack_to_one`, `refine_node_types_from_generators`)
перенесена в ``gridstate.topology`` (Фаза 3 target-architecture; раньше жила в
``power_system.topology``). Используется напрямую::

    from gridstate.topology import (
        disable_orphan_branches,
        disable_disconnected_components,
        disable_isolated_nodes,
        refine_slack_to_one,
        refine_node_types_from_generators,
    )

SE-специфичные шаги (псевдо-измерения, чтение properties входного формата) — здесь.
"""

from gridstate.preprocessing.chain_voltage import chain_pseudo_voltage_through_tap_links
from gridstate.preprocessing.dead_gen_nodes import disable_dead_generator_nodes
from gridstate.preprocessing.mirror_voltage import mirror_voltage_through_unit_tap_links
from gridstate.preprocessing.node_props import (
    extract_boundary_node_ids_from_model,
    extract_node_load_props_from_model,
)
from gridstate.preprocessing.one_sided import (
    classify_branch_connectivity,
    fold_one_sided_branches,
)
from gridstate.preprocessing.pseudo_measurements import add_pseudo_measurements
from gridstate.preprocessing.synth_injection import (
    synthesize_block_bus_injection_from_branch_xml,
    synthesize_node_injection_from_branch_flows,
)


__all__ = [
    "add_pseudo_measurements",
    "chain_pseudo_voltage_through_tap_links",
    "classify_branch_connectivity",
    "disable_dead_generator_nodes",
    "extract_boundary_node_ids_from_model",
    "extract_node_load_props_from_model",
    "fold_one_sided_branches",
    "mirror_voltage_through_unit_tap_links",
    "synthesize_block_bus_injection_from_branch_xml",
    "synthesize_node_injection_from_branch_flows",
]
