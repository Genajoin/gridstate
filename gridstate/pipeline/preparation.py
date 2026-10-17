"""Pre-solve step adapters: network derivations, telemetry, measurement preparation.

Each adapter takes the run context and returns a stats dict. The adapters only
bind configuration values and derived plans to the library functions; the
order they run in is declared by the registry (:mod:`gridstate.pipeline.steps`).
"""

from __future__ import annotations

from gridstate.pipeline.context import _Ctx, _missing_plan
from gridstate.preprocessing import (
    add_pseudo_measurements,
    mirror_voltage_through_unit_tap_links,
)
from gridstate.telemetry import (
    aggregate_generators_to_node,
    apply_flow_sigma_floor,
    apply_generator_status_from_node,
    apply_reactors_to_node_shunt,
    apply_voltage_meas_calibration_for_gen_nodes,
    apply_voltage_range_filter,
    deactivate_orphan_measurements,
    normalize_breaker_reactance,
)
from gridstate.telemetry.apply_resolved import apply_materialize_resolved, apply_telemetry_resolved
from gridstate.telemetry.load_only_injection import (
    generation_only_injection_nodes,
    load_only_injection_nodes,
    relax_generation_min_to_injection,
    release_load_only_injections,
    widen_generation_only_injections,
)
from gridstate.telemetry.on_line import apply_topology_resolved
from gridstate.telemetry.rpn import apply_rpn_resolved
from gridstate.telemetry.voltage_nominal import apply_voltage_nominal_resolved
from gridstate.topology import (
    disable_disconnected_components,
    disable_isolated_nodes,
    disable_orphan_branches,
    refine_node_types_from_generators,
    refine_slack_to_one,
)


# --- network derivations ---


def _s_normalize_breakers(ctx: _Ctx) -> dict:
    return dict(normalize_breaker_reactance(ctx.model) or {})


def _s_voltage_nominal(ctx: _Ctx) -> dict:
    # Apply the ready Vnom plan (computed outside the core).
    if ctx.derived is None or ctx.derived.voltage_nominal is None:
        raise _missing_plan("voltage_nominal", "voltage_nominal")
    apply_voltage_nominal_resolved(ctx.model, ctx.derived.voltage_nominal)
    return {}


def _s_topology(ctx: _Ctx) -> dict:
    # Apply the ready ON_LINE status plan (computed outside the core).
    if ctx.derived is None or ctx.derived.topology_resolved is None:
        raise _missing_plan("topology", "topology_resolved")
    return dict(apply_topology_resolved(ctx.model, ctx.derived.topology_resolved) or {})


def _s_rpn(ctx: _Ctx) -> dict:
    return dict(apply_rpn_resolved(ctx.model) or {})


def _s_reactors(ctx: _Ctx) -> dict:
    return dict(apply_reactors_to_node_shunt(ctx.model) or {})


def _s_refine_slack(ctx: _Ctx) -> dict:
    return dict(refine_slack_to_one(ctx.model) or {})


def _s_refine_node_types(ctx: _Ctx) -> dict:
    return dict(refine_node_types_from_generators(ctx.model) or {})


def _s_disable_orphan_branches(ctx: _Ctx) -> dict:
    # The repeat after disconnected_components is done by the paired step below.
    return dict(disable_orphan_branches(ctx.model) or {})


def _s_disable_disconnected(ctx: _Ctx) -> dict:
    stats = dict(disable_disconnected_components(ctx.model) or {})
    # Switch off again the branches incident to the nodes just disabled.
    if ctx.cfg.disable_orphan_branches:
        recheck = dict(disable_orphan_branches(ctx.model) or {})
        stats["orphan_recheck_disabled"] = recheck.get("disabled", 0)
    return stats


def _s_disable_isolated(ctx: _Ctx) -> dict:
    return dict(disable_isolated_nodes(ctx.model) or {})


def _s_generator_status(ctx: _Ctx) -> dict:
    return dict(apply_generator_status_from_node(ctx.model) or {})


# --- telemetry and measurement preparation ---


def _s_telemetry(ctx: _Ctx) -> dict:
    # Apply the ready z-vector (computed outside the core). Statuses of the
    # input measurements (they arrive with status=True) are cleared by the
    # kernel itself (_apply_telemetry_on_arrays sets arr["status"]=False
    # before activating).
    if (
        ctx.derived is None
        or ctx.derived.telemetry_resolved is None
        or ctx.derived.telemetry_arg_keys is None
    ):
        raise _missing_plan("telemetry", "telemetry_resolved", "telemetry_arg_keys")
    return dict(
        apply_telemetry_resolved(
            ctx.model,
            ctx.derived.telemetry_resolved,
            ctx.derived.telemetry_arg_keys,
            total_args=ctx.derived.telemetry_total_args,
            v_sigma2_scale_by_node=ctx.derived.v_sigma2_scale,
            flow_sigma2_scale_by_branch=ctx.derived.flow_sigma2_scale,
            injection_roles=ctx.cfg.ipm_injection_roles,
            telemetry_bounds=ctx.derived.telemetry_bounds,
        )
        or {}
    )


def _s_voltage_range_filter(ctx: _Ctx) -> dict:
    return dict(
        apply_voltage_range_filter(
            ctx.model,
            min_voltage_nominal_kv=ctx.cfg.voltage_filter_min_nominal_kv,
            deactivate_nonphysical=ctx.cfg.voltage_filter_deactivate_nonphysical,
            nonphysical_lower_factor=ctx.cfg.voltage_filter_nonphysical_lower_factor,
            nonphysical_upper_factor=ctx.cfg.voltage_filter_nonphysical_upper_factor,
        )
        or {}
    )


def _s_aggregate_generators(ctx: _Ctx) -> dict:
    return dict(aggregate_generators_to_node(ctx.model) or {})


def _s_release_load_only_injections(ctx: _Ctx) -> dict:
    if ctx.cfg.ipm_injection_roles:
        # The parts are measured separately: a partial net injection does not arise.
        return {"skipped_injection_roles": 1}
    if (
        ctx.derived is None
        or ctx.derived.telemetry_resolved is None
        or ctx.derived.telemetry_arg_keys is None
    ):
        raise _missing_plan(
            "release_load_only_injections", "telemetry_resolved", "telemetry_arg_keys"
        )
    load_only = load_only_injection_nodes(
        ctx.derived.telemetry_resolved, ctx.derived.telemetry_arg_keys
    )
    stats, ctx.released_injections = release_load_only_injections(ctx.model, load_only)
    return stats


def _s_relax_generation_min_to_injection(ctx: _Ctx) -> dict:
    if (
        ctx.derived is None
        or ctx.derived.telemetry_resolved is None
        or ctx.derived.telemetry_arg_keys is None
    ):
        raise _missing_plan(
            "relax_generation_min_to_injection", "telemetry_resolved", "telemetry_arg_keys"
        )
    gen_only = generation_only_injection_nodes(
        ctx.derived.telemetry_resolved, ctx.derived.telemetry_arg_keys
    )
    return relax_generation_min_to_injection(ctx.model, gen_only)


def _s_widen_generation_only_injections(ctx: _Ctx) -> dict:
    if ctx.cfg.ipm_injection_roles:
        # The parts are measured separately: a partial net injection does not arise.
        return {"skipped_injection_roles": 1}
    if (
        ctx.derived is None
        or ctx.derived.telemetry_resolved is None
        or ctx.derived.telemetry_arg_keys is None
    ):
        raise _missing_plan(
            "widen_generation_only_injections", "telemetry_resolved", "telemetry_arg_keys"
        )
    gen_only = generation_only_injection_nodes(
        ctx.derived.telemetry_resolved, ctx.derived.telemetry_arg_keys
    )
    return widen_generation_only_injections(ctx.model, gen_only)


def _s_gen_v_calibration(ctx: _Ctx) -> dict:
    return dict(
        apply_voltage_meas_calibration_for_gen_nodes(
            ctx.model,
            mode=ctx.cfg.gen_v_calibration_mode,  # type: ignore[arg-type]
        )
        or {}
    )


def _s_deactivate_orphan_measurements(ctx: _Ctx) -> dict:
    return dict(deactivate_orphan_measurements(ctx.model) or {})


def _s_mirror_voltage(ctx: _Ctx) -> dict:
    return dict(mirror_voltage_through_unit_tap_links(ctx.model) or {})


def _s_materialize(ctx: _Ctx) -> dict:
    # Apply the observed nodal operating point computed outside the core (the
    # producer of the input resolves it, including the sign convention).
    if ctx.derived is None or ctx.derived.materialize_obs is None:
        return {"skipped": "нет плана материализации (нужны наблюдаемые инжекции)"}
    apply_materialize_resolved(ctx.model, ctx.derived.materialize_obs)
    return {}


def _s_add_pseudo(ctx: _Ctx) -> dict:
    return dict(
        add_pseudo_measurements(
            ctx.model,
            unobservable_v_sigma_frac=ctx.cfg.unobservable_v_sigma_frac,
            unobservable_v_min_vm_deviation=ctx.cfg.unobservable_v_min_vm_deviation,
            unmeasured_gen_p_nodes=ctx.released_injections.get("P", frozenset()),
            unmeasured_gen_q_nodes=ctx.released_injections.get("Q", frozenset()),
            loosen_unknown_load=ctx.cfg.loosen_unknown_load_pseudo,
        )
        or {}
    )


def _s_flow_sigma_floor(ctx: _Ctx) -> dict:
    # σ floor of real flow measurements from the channel full scale. Runs AFTER
    # add_pseudo: the is_pseudo==0 selector guarantees that the pseudo priors
    # are untouched, and the final variances of the real flow measurements
    # match the validated A/B prototype.
    frac = ctx.cfg.flow_sigma_floor_kv_frac
    if frac is None or frac <= 0:
        return {"skipped": "выключено (flow_sigma_floor_kv_frac=None)"}
    return dict(apply_flow_sigma_floor(ctx.model, kv_frac=float(frac)) or {})
