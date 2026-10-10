"""Tests of the library wrapper ``gridstate.pipeline``.

* manifest consistency (structure, JSON serialisability, toggle <-> config);
* production defaults of ``PipelineConfig``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields

import pytest

from gridstate import pipeline as P  # noqa: N812


# ---------------------------------------------------------------------------
# Manifest / registry (no data)
# ---------------------------------------------------------------------------


def test_manifest_is_json_serializable():
    man = P.manifest()
    json.dumps(man)  # must not raise
    assert set(man) == {"steps", "params", "groups"}
    assert man["steps"] and man["params"] and man["groups"]


def test_manifest_steps_have_required_fields():
    for s in P.manifest()["steps"]:
        assert set(s) >= {
            "name",
            "title",
            "group",
            "description",
            "optional",
            "toggle",
            "default_enabled",
            "needs_derived",
        }
        # optional <=> there is a toggle field
        assert s["optional"] == (s["toggle"] is not None)


def test_manifest_params_have_required_fields():
    for p in P.manifest()["params"]:
        assert set(p) >= {"name", "kind", "type", "default", "control", "group", "label"}
        assert p["type"] in {"bool", "int", "float", "str"}


def test_every_step_toggle_references_real_config_field():
    cfg_fields = {f.name for f in fields(P.PipelineConfig)}
    for step in P.STEPS:
        if step.toggle is not None:
            assert step.toggle in cfg_fields, f"{step.name}: no field {step.toggle}"


def test_manifest_default_enabled_matches_config():
    cfg = P.PipelineConfig()
    for s in P.manifest()["steps"]:
        if s["toggle"] is not None:
            assert s["default_enabled"] == getattr(cfg, s["toggle"])
        else:
            assert s["default_enabled"] is True


def test_manifest_param_defaults_match_config():
    cfg = P.PipelineConfig()
    for p in P.manifest()["params"]:
        assert p["default"] == getattr(cfg, p["name"])


def test_manifest_config_none_class_and_instance_identical():
    base = P.manifest()
    assert P.manifest(None) == base
    assert P.manifest(P.PipelineConfig) == base
    assert P.manifest(P.PipelineConfig()) == base
    assert [p["name"] for p in base["params"]] == [
        f.name for f in fields(P.PipelineConfig) if f.metadata
    ]


def _by_name(items: list[dict]) -> dict[str, dict]:
    return {i["name"]: i for i in items}


def test_manifest_subclass_defaults_reflected():
    @dataclass
    class Custom(P.PipelineConfig):
        normalize_breakers: bool = False
        voltage_filter_min_nominal_kv: float = 35.0

    base = _by_name(P.manifest()["params"])
    man = P.manifest(Custom)
    params = _by_name(man["params"])
    assert params["normalize_breakers"]["default"] is False
    assert params["voltage_filter_min_nominal_kv"]["default"] == 35.0
    # Everything else (metadata included) is untouched.
    assert base["normalize_breakers"]["default"] is True
    for name, entry in params.items():
        if name not in ("normalize_breakers", "voltage_filter_min_nominal_kv"):
            assert entry == base[name]
    step = next(s for s in man["steps"] if s["toggle"] == "normalize_breakers")
    assert step["default_enabled"] is False
    assert json.loads(json.dumps(man)) == man


def test_manifest_subclass_extra_field_with_metadata():
    @dataclass
    class Custom(P.PipelineConfig):
        extra_knob: float = P._param(
            1.5, group="Custom group", label="Extra", control="number", min=0.0, max=3.0, help="h"
        )

    params = _by_name(P.manifest(Custom)["params"])
    assert params["extra_knob"]["default"] == 1.5
    assert params["extra_knob"]["type"] == "float"
    assert params["extra_knob"]["max"] == 3.0
    assert "extra_knob" not in _by_name(P.manifest()["params"])


def test_manifest_instance_values_reflected():
    cfg = P.PipelineConfig(apply_rpn=False, voltage_filter_min_nominal_kv=0.0)
    man = P.manifest(cfg)
    params = _by_name(man["params"])
    assert params["voltage_filter_min_nominal_kv"]["default"] == 0.0
    assert params["apply_rpn"]["default"] is False
    step = next(s for s in man["steps"] if s["toggle"] == "apply_rpn")
    assert step["default_enabled"] is False
    # The instance is not mutated and library defaults are unaffected.
    assert cfg.apply_rpn is False
    assert _by_name(P.manifest()["params"])["apply_rpn"]["default"] is True


def test_manifest_rejects_invalid_config():
    with pytest.raises(TypeError):
        P.manifest(dict)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        P.manifest({"apply_rpn": False})  # type: ignore[arg-type]


def test_groups_cover_all_entries():
    man = P.manifest()
    groups = set(man["groups"])
    for s in man["steps"]:
        assert s["group"] in groups
    for p in man["params"]:
        assert p["group"] in groups


def test_step_names_unique():
    names = [s.name for s in P.STEPS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Production defaults (single source of truth)
# ---------------------------------------------------------------------------


def test_production_defaults():
    cfg = P.PipelineConfig()
    assert cfg.algorithm == "wls"
    assert cfg.materialize is True
    assert cfg.anti_overshoot is True
    assert cfg.anti_overshoot_ceiling == 1.15
    assert cfg.normalize_breakers is True
    assert cfg.max_iterations == 80


def test_effective_huber_c_auto():
    assert P._effective_huber_c(P.PipelineConfig(algorithm="wls")) == 1.5
    assert P._effective_huber_c(P.PipelineConfig(algorithm="ipm")) == 2.0
    assert P._effective_huber_c(P.PipelineConfig(huber_c=5.0)) == 5.0


# ---------------------------------------------------------------------------
# Config <-> derived consistency guard ("cfg twice")
# ---------------------------------------------------------------------------


def _full_derived() -> P.DerivedInputs:
    from gridstate.contract.derived import DerivedInputs

    return DerivedInputs(
        topology_resolved=[],
        telemetry_resolved={},
        telemetry_arg_keys=[],
        telemetry_total_args=0,
    )


def test_derived_consistency_ok_on_matched_config():
    P._check_derived_consistency(P.PipelineConfig(), _full_derived())


def test_derived_consistency_rejects_missing_topology_plan():
    import pytest

    d = _full_derived()
    d.topology_resolved = None
    with pytest.raises(ValueError, match="topology"):
        P._check_derived_consistency(P.PipelineConfig(apply_topology=True), d)
    # a disabled toggle makes a missing plan legal
    P._check_derived_consistency(P.PipelineConfig(apply_topology=False), d)


def test_derived_consistency_rejects_missing_telemetry_plan():
    import pytest

    d = _full_derived()
    d.telemetry_resolved = None
    with pytest.raises(ValueError, match="telemetry"):
        P._check_derived_consistency(P.PipelineConfig(), d)


def test_derived_consistency_network_only_skips_telemetry():
    # prepare_network runs only network=True steps: telemetry is not one of them,
    # so its plan is not required in this mode.
    d = _full_derived()
    d.telemetry_resolved = None
    P._check_derived_consistency(P.PipelineConfig(), d, network_only=True)
