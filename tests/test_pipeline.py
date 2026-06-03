"""Тесты библиотечной обёртки ``gridstate.pipeline``.

* manifest-консистентность (структура, JSON-сериализуемость, toggle↔config);
* production-дефолты ``PipelineConfig``.
"""

from __future__ import annotations

import json
from dataclasses import fields

from gridstate import pipeline as P  # noqa: N812


# ---------------------------------------------------------------------------
# Манифест / реестр (без данных)
# ---------------------------------------------------------------------------


def test_manifest_is_json_serializable():
    man = P.manifest()
    json.dumps(man)  # не должно бросить
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
            "needs_xml",
        }
        # optional ⟺ есть toggle-поле
        assert s["optional"] == (s["toggle"] is not None)


def test_manifest_params_have_required_fields():
    for p in P.manifest()["params"]:
        assert set(p) >= {"name", "kind", "type", "default", "control", "group", "label"}
        assert p["type"] in {"bool", "int", "float", "str"}


def test_every_step_toggle_references_real_config_field():
    cfg_fields = {f.name for f in fields(P.PipelineConfig)}
    for step in P.STEPS:
        if step.toggle is not None:
            assert step.toggle in cfg_fields, f"{step.name}: нет поля {step.toggle}"


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
# Production-дефолты (единственный источник истины)
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
