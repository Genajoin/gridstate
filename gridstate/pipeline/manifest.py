"""Declarative, JSON-serializable description of the pipeline for a UI."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from gridstate.pipeline.config import PipelineConfig, _effective_metadata
from gridstate.pipeline.steps import STEPS


def manifest(config: PipelineConfig | type[PipelineConfig] | None = None) -> dict:
    """Declarative description of the pipeline for a UI.

    Returns a JSON-serializable dict::

        {
          "steps":  [{name, title, group, description, optional, toggle,
                      default_enabled, needs_derived}, ...],   # execution order
          "params": [{name, type, default, control, group, label, help,
                      min?, max?, choices?, depends?}, ...],
          "groups": [...],                                     # section order
        }

    A UI builds its form by iterating ``params`` (controls) and ``steps`` (the
    toggle list), grouped by ``group``. It takes the defaults from here instead
    of hard-coding them.

    ``depends`` maps field names to the values under which the control is shown;
    every entry must match. A scalar value matches by equality, a list matches
    any of its values: ``{"algorithm": ["ipm", "ipm_pd"]}`` shows the control
    for both IPM solvers.

    ``config`` selects the source of defaults and of the iterated fields:

    * ``None`` (default) -- ``PipelineConfig()``, the library defaults;
    * a ``PipelineConfig`` subclass -- instantiated without arguments, so a subclass
      that overrides defaults (or adds fields with metadata) is described as is;
    * a ``PipelineConfig`` instance -- its current values are reported as defaults.

    A UI that fills its form from the manifest then does not overwrite the defaults
    of a downstream configuration with the library ones. A subclass field redeclared
    with a bare default (no metadata) keeps the base field's UI metadata.
    """
    if config is None:
        cfg_defaults = PipelineConfig()
    elif isinstance(config, type):
        if not issubclass(config, PipelineConfig):
            raise TypeError(f"config must be a PipelineConfig subclass or instance, got {config!r}")
        cfg_defaults = config()
    elif isinstance(config, PipelineConfig):
        cfg_defaults = config
    else:
        raise TypeError(f"config must be a PipelineConfig subclass or instance, got {config!r}")

    params: list[dict] = []
    groups_order: list[str] = []
    for f in fields(cfg_defaults):
        # A subclass that re-declares a field with a bare default drops its metadata;
        # fall back to the base declaration so the control does not disappear.
        md = _effective_metadata(f)
        if not md:
            continue
        grp = md.get("group", "")
        if grp and grp not in groups_order:
            groups_order.append(grp)
        default = getattr(cfg_defaults, f.name)
        entry = {
            "name": f.name,
            "kind": md.get("kind", "param"),
            "type": _type_name(f.type),
            "default": default,
            "control": md.get("control"),
            "group": grp,
            "label": md.get("label", f.name),
            "help": md.get("help", ""),
        }
        for k in ("min", "max", "choices", "depends"):
            if k in md:
                entry[k] = md[k]
        params.append(entry)

    steps_out: list[dict] = []
    for step in STEPS:
        if step.group and step.group not in groups_order:
            groups_order.append(step.group)
        steps_out.append(
            {
                "name": step.name,
                "title": step.title,
                "group": step.group,
                "description": step.description,
                "optional": step.toggle is not None,
                "toggle": step.toggle,
                "default_enabled": (
                    getattr(cfg_defaults, step.toggle) if step.toggle is not None else True
                ),
                "needs_derived": step.needs_derived,
            }
        )

    return {"steps": steps_out, "params": params, "groups": groups_order}


def _type_name(tp: Any) -> str:
    """Field type name for the UI (str/int/float/bool; float|None → 'float')."""
    s = str(tp)
    for name in ("bool", "int", "float", "str"):
        if name in s:
            return name
    return "str"
