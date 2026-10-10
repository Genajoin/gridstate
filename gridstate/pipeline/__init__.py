"""Library wrapper of the SE pipeline.

One entry point for external consumers (UI, CLI) instead of duplicating the
orchestration in each of them: the input is a working layer ``model`` (plus
optional precomputed numeric plans ``derived``) and a :class:`PipelineConfig`.

Modules of the package:

* :mod:`~gridstate.pipeline.config` -- :class:`PipelineConfig`, a dataclass
  with ALL the knobs and their UI metadata; **its defaults are the single
  source of truth** (gridstate does not depend on pydantic);
* :mod:`~gridstate.pipeline.context` -- the run context and the :class:`Step`
  descriptor;
* :mod:`~gridstate.pipeline.preparation` -- pre-solve step adapters (network
  derivations, telemetry, measurement preparation);
* :mod:`~gridstate.pipeline.passes` -- solver calls: the main solve and the
  post-solve warm re-solve passes;
* :mod:`~gridstate.pipeline.steps` -- :data:`STEPS`, the ordered step registry
  (name, title, group, description, toggle field);
* :mod:`~gridstate.pipeline.runner` -- :func:`run` executes the registry in
  order, :func:`prepare_network` its network subset; progress events;
* :mod:`~gridstate.pipeline.manifest` -- :func:`manifest`, a JSON-serializable
  description (steps + parameters with defaults, types, bounds,
  label/help/group). A UI builds its form from it, without hard-coding the list
  of functions or defaults.

Progress is reported through the ``on_event(event: dict)`` callback (the
protocol is compatible with UI streaming events): ``step_start`` /
``step_done`` / ``step_skipped`` / ``step_error``. gridstate does not depend on
a web framework; the consumer wraps the events into its own transport (NDJSON
and the like).

Example::

    from gridstate.contract.serialize import load_se_input_npz
    from gridstate.contract.runtime import run as contract_run
    from gridstate.pipeline import PipelineConfig, manifest

    se_input = load_se_input_npz("model.npz")
    result = contract_run(se_input, config=PipelineConfig())
    # UI:
    schema = manifest()   # {"steps": [...], "params": {...}, "groups": [...]}

Every name defined by the former single ``gridstate.pipeline`` module is
re-exported here (private helpers included, for existing callers and tests).
The library functions the steps call are not: they are looked up at call time
in the module that uses them, so a test intercepting the solver or a
post-solve classifier patches :mod:`gridstate.pipeline.passes` (for example
``gridstate.pipeline.passes.estimate``); patching a name that is not defined
here fails loudly instead of silently having no effect.
"""

from __future__ import annotations

from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline.config import (
    _G_CASCADE,
    _G_EST,
    _G_INPUT,
    _G_IPM,
    _G_MODE,
    _G_POST,
    _G_PSEUDO,
    PipelineConfig,
    _param,
    _toggle,
)
from gridstate.pipeline.context import Step, _Ctx
from gridstate.pipeline.manifest import _type_name, manifest
from gridstate.pipeline.passes import (
    _MERGED_SKIP,
    _effective_huber_c,
    _estimate_kwargs,
    _ipm_kwargs,
    _refine_two_pass,
    _s_anti_overshoot,
    _s_bad_data_repass,
    _s_estimate,
    _s_refine_merged,
    _s_v_mirror,
    _s_v_mirror_chain,
    _s_v_refine,
    _warm_resolve,
)
from gridstate.pipeline.preparation import (
    _s_add_pseudo,
    _s_aggregate_generators,
    _s_deactivate_orphan_measurements,
    _s_disable_disconnected,
    _s_disable_isolated,
    _s_disable_orphan_branches,
    _s_flow_sigma_floor,
    _s_gen_v_calibration,
    _s_generator_status,
    _s_materialize,
    _s_mirror_voltage,
    _s_normalize_breakers,
    _s_reactors,
    _s_refine_node_types,
    _s_refine_slack,
    _s_relax_generation_min_to_injection,
    _s_release_load_only_injections,
    _s_rpn,
    _s_telemetry,
    _s_topology,
    _s_voltage_nominal,
    _s_voltage_range_filter,
    _s_widen_generation_only_injections,
)
from gridstate.pipeline.runner import (
    _REQUIRED_DERIVED_PLANS,
    _build_working,
    _check_derived_consistency,
    _emit,
    _execute_step,
    _seed_warm_start,
    prepare_network,
    run,
)
from gridstate.pipeline.steps import STEPS


__all__ = [
    "STEPS",
    "DerivedInputs",
    "PipelineConfig",
    "Step",
    "manifest",
    "prepare_network",
    "run",
]

# Compatibility: private helpers of the former single module, imported from
# ``gridstate.pipeline`` by existing callers and tests.
__all__ += [
    "_G_CASCADE",
    "_G_EST",
    "_G_INPUT",
    "_G_IPM",
    "_G_MODE",
    "_G_POST",
    "_G_PSEUDO",
    "_MERGED_SKIP",
    "_REQUIRED_DERIVED_PLANS",
    "_Ctx",
    "_build_working",
    "_check_derived_consistency",
    "_effective_huber_c",
    "_emit",
    "_estimate_kwargs",
    "_execute_step",
    "_ipm_kwargs",
    "_param",
    "_refine_two_pass",
    "_s_add_pseudo",
    "_s_aggregate_generators",
    "_s_anti_overshoot",
    "_s_bad_data_repass",
    "_s_deactivate_orphan_measurements",
    "_s_disable_disconnected",
    "_s_disable_isolated",
    "_s_disable_orphan_branches",
    "_s_estimate",
    "_s_flow_sigma_floor",
    "_s_gen_v_calibration",
    "_s_generator_status",
    "_s_materialize",
    "_s_mirror_voltage",
    "_s_normalize_breakers",
    "_s_reactors",
    "_s_refine_merged",
    "_s_refine_node_types",
    "_s_refine_slack",
    "_s_relax_generation_min_to_injection",
    "_s_release_load_only_injections",
    "_s_rpn",
    "_s_telemetry",
    "_s_topology",
    "_s_v_mirror",
    "_s_v_mirror_chain",
    "_s_v_refine",
    "_s_voltage_nominal",
    "_s_voltage_range_filter",
    "_s_widen_generation_only_injections",
    "_seed_warm_start",
    "_toggle",
    "_type_name",
    "_warm_resolve",
]
