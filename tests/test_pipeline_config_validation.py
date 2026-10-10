"""``PipelineConfig`` validates its values against the ``choices``/``min``/``max`` metadata."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import pytest

from gridstate.pipeline import PipelineConfig, prepare_network, run
from gridstate.pipeline.config import _G_POST, _param


def test_defaults_are_valid() -> None:
    PipelineConfig().validate()


def test_every_metadata_default_is_within_its_bounds() -> None:
    cfg = PipelineConfig()
    for f in fields(cfg):
        md = f.metadata
        value = getattr(cfg, f.name)
        if value is None:
            continue
        if "choices" in md:
            assert value in md["choices"], f.name
        if "min" in md:
            assert value >= md["min"], f.name
        if "max" in md:
            assert value <= md["max"], f.name


@dataclass
class _DeploymentConfig(PipelineConfig):
    """A downstream subclass: bare redeclarations change defaults only."""

    algorithm: str = "ipm"
    flow_sigma_floor_kv_frac: float | None = 0.010
    bad_data: bool = True
    v_refine: bool = True
    v_refine_factor: float = 0.6
    v_mirror: bool = True
    v_mirror_min_lift: float = 0.005
    v_mirror_cross_at: bool = True
    v_mirror_chain: bool = True
    refine_merge: bool = True
    voltage_filter_min_nominal_kv: float = 0.0
    voltage_filter_deactivate_nonphysical: bool = True
    relax_generation_min_to_injection: bool = True
    ipm_prior_sigma2_bus_equiv_pu: float = 1.0
    ipm_boxed_balance_weight_factor: float = 1.0
    anti_overshoot: bool = False
    reconcile_respect_bounds: bool = True
    # a field of the subclass only, without metadata: not checked
    extra_weights: dict[int, float] | None = None


def test_subclass_with_changed_defaults_is_valid() -> None:
    cfg = _DeploymentConfig()
    assert cfg.algorithm == "ipm"
    cfg.validate()


def test_bare_redeclaration_is_checked_against_the_base_metadata() -> None:
    with pytest.raises(ValueError, match=r"v_refine_factor=2\.0 is above the maximum 1\.0"):
        _DeploymentConfig(v_refine_factor=2.0)
    with pytest.raises(ValueError, match="algorithm='IPM' is not one of"):
        _DeploymentConfig(algorithm="IPM")


def test_out_of_range_subclass_default_is_rejected() -> None:
    @dataclass
    class _Bad(PipelineConfig):
        tolerance: float = 0.0

    with pytest.raises(ValueError, match=r"tolerance=0\.0 is below the minimum 1e-08"):
        _Bad()


def test_subclass_field_with_metadata_is_checked() -> None:
    @dataclass
    class _Extra(PipelineConfig):
        extra_knob: float = _param(1.0, group=_G_POST, label="x", control="number", max=2.0)

    _Extra()
    with pytest.raises(ValueError, match=r"extra_knob=3\.0"):
        _Extra(extra_knob=3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"algorithm": "IPM"}, "algorithm='IPM' is not one of ['wls', 'ipm']"),
        ({"init": "dc"}, "init='dc' is not one of"),
        ({"kkt_solver": "umfpack"}, "kkt_solver='umfpack' is not one of"),
        (
            {"ipm_trust_region_scope": "box"},
            "ipm_trust_region_scope='box' is not one of ['all', 'state']",
        ),
        ({"gen_v_calibration_mode": "loose"}, "gen_v_calibration_mode='loose'"),
        ({"tolerance": 0.0}, "tolerance=0.0 is below the minimum"),
        ({"max_iterations": 0}, "max_iterations=0 is below the minimum 1"),
        ({"max_iterations": 501}, "max_iterations=501 is above the maximum 500"),
        ({"anti_overshoot_ceiling": 0.9}, "anti_overshoot_ceiling=0.9 is below"),
        ({"flow_sigma_floor_kv_frac": 2.0}, "flow_sigma_floor_kv_frac=2.0 is above"),
        ({"tolerance": float("nan")}, "tolerance=nan is not a number"),
        ({"tolerance": "1e-3"}, "tolerance='1e-3' is not a number"),
        ({"max_iterations": True}, "max_iterations=True is not a number"),
        ({"tolerance": None}, "tolerance=None is not allowed"),
        ({"algorithm": None}, "algorithm=None is not allowed"),
    ],
)
def test_invalid_value_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match="invalid PipelineConfig") as exc:
        PipelineConfig(**kwargs)
    assert message in str(exc.value)


def test_optional_fields_accept_none() -> None:
    PipelineConfig(huber_c=None, flow_sigma_floor_kv_frac=None)


def test_all_problems_are_reported_together() -> None:
    with pytest.raises(ValueError) as exc:
        PipelineConfig(algorithm="x", tolerance=-1.0, top_residuals_n=1000)
    text = str(exc.value)
    assert "algorithm='x'" in text
    assert "tolerance=-1.0" in text
    assert "top_residuals_n=1000" in text


def test_ipm_trust_region_scope_default_and_forwarding() -> None:
    from gridstate.pipeline.passes import _estimate_kwargs

    assert PipelineConfig().ipm_trust_region_scope == "state"
    assert _estimate_kwargs(PipelineConfig(algorithm="ipm"), init="flat")["trust_region_scope"] == (
        "state"
    )
    cfg = PipelineConfig(algorithm="ipm", ipm_trust_region_scope="all")
    assert _estimate_kwargs(cfg, init="flat")["trust_region_scope"] == "all"
    # WLS never sees the IPM knob.
    assert "trust_region_scope" not in _estimate_kwargs(PipelineConfig(), init="flat")


def test_replace_validates() -> None:
    with pytest.raises(ValueError, match=r"bad_data_threshold=0\.5"):
        replace(PipelineConfig(), bad_data_threshold=0.5)


def test_run_revalidates_an_edited_config() -> None:
    cfg = PipelineConfig()
    cfg.algorithm = "lav"
    # the check comes before the input is read, so no model is needed
    with pytest.raises(ValueError, match="algorithm='lav'"):
        run(None, config=cfg)
    with pytest.raises(ValueError, match="algorithm='lav'"):
        prepare_network(None, config=cfg)
