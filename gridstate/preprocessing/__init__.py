"""Model preparation for WLS: pseudo-measurements and input-format properties.

Topology cleanup (`disable_orphan_branches`, `disable_disconnected_components`,
`disable_isolated_nodes`, `refine_slack_to_one`, `refine_node_types_from_generators`)
lives in ``gridstate.topology``. Use it directly::

    from gridstate.topology import (
        disable_orphan_branches,
        disable_disconnected_components,
        disable_isolated_nodes,
        refine_slack_to_one,
        refine_node_types_from_generators,
    )

SE-specific steps (pseudo-measurements, reading input-format properties) live here.
"""

from gridstate.preprocessing.mirror_voltage import mirror_voltage_through_unit_tap_links
from gridstate.preprocessing.node_props import extract_node_load_props_from_model
from gridstate.preprocessing.pseudo_measurements import add_pseudo_measurements


__all__ = [
    "add_pseudo_measurements",
    "extract_node_load_props_from_model",
    "mirror_voltage_through_unit_tap_links",
]
