"""Golden regression harness: record and compare full SE pipeline snapshots.

A *snapshot* captures everything a pipeline run produces for one input and one
configuration:

* every column of the ``SEOutput`` tables (``nodes`` / ``branches`` /
  ``measurements``) plus ``v_pu`` / ``delta_rad``;
* every column of every table of the solved working model (``result.model``):
  V/delta, flows, per-measurement ``estimated_si`` / ``residual``, pseudo
  measurements with their ids, statuses and variances after the post-solve
  passes;
* the quality summary (chi2, worst residuals, worst imbalances, observability
  warnings);
* ``success`` / ``iterations`` / ``objective_value`` / ``algorithm`` /
  ``convergence_status``;
* the ordered pipeline event sequence (type, step name, step stats). Timings
  are dropped; free-text strings (skip reasons, prose stats) are replaced by a
  ``"<text>"`` marker, so a reworded message does not count as a change while a
  step that stops or starts being skipped does.

Snapshots of one input are stored as ``<input>.npz`` (arrays, keyed
``<config>/<array>``) plus ``<input>.json`` (scalars and events), with a
``manifest.json`` holding the environment (gridstate version, git sha,
numpy/scipy/cvxopt versions, KKT backend), the configurations and input
fingerprints.

Comparison has two modes: *strict* (``np.array_equal`` including NaN positions,
exact scalars and events) for same-machine refactoring checks, and *tolerant*
(element-wise ``|ref - cand| <= atol + rtol * |ref|`` on floats, exact on
integers / booleans / strings / event sequence) for cross-platform CI.

Inputs
------

The committed IEEE fixtures (``tests/test_data/ieee/caseNN.npz``) are fully
measured and carry no box bounds, so on them most post-solve passes are
no-ops. Each case is therefore recorded twice: as-is (``caseNN``) and as a
deterministic in-memory *degraded* variant (``caseNN_degraded``, see
:func:`degrade`) that makes every pass of the full IPM configuration do real
work: finite load/generation boxes, |V| readings only at generator nodes,
blind nodes, a sign-flipped flow, a gross reactive injection error and a wrong
shunt. Any directory of contract ``.npz`` inputs can be recorded too (kept
outside the repository).

The command line front end is ``tools/golden.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gridstate.contract import SE_INPUT, SEInput, SEOutput
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.pipeline import PipelineConfig
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
IEEE_DIR = REPO_ROOT / "tests" / "test_data" / "ieee"
REFERENCE_DIR = REPO_ROOT / "tests" / "test_data" / "golden"

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Tolerant-mode defaults, calibrated empirically (evidence in the
# tests/test_golden.py docstring). atol is in the native units of each column
# (kV, MW, Mvar, A, rad, ...): 1e-6 of any of them is physically meaningless,
# yet orders of magnitude below what a changed discrete decision (bad-data or
# refine plan, iteration count) produces.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-6

# Per-(input, config) widening of the tolerant mode, with the reason. Strict
# mode ignores it.
TOLERANCE_SCALE: dict[tuple[str, str], tuple[float, str]] = {
    ("case118", "warm_start"): (
        1000.0,
        "IPM warm-started from the WLS point takes a roundoff-sensitive path on "
        "this case: the CHOLMOD and scipy backends agree to 1e-15 on the start "
        "point, yet the first IPM objective differs by 2e-4 (relative) and final "
        "flows by up to 1.8e-4 (absolute); every other input/config stays below "
        "6e-8 (relative)",
    ),
}

IEEE_CASES = ("case14", "case30", "case118")
DEGRADED_SUFFIX = "_degraded"
IEEE_INPUTS = tuple(name for case in IEEE_CASES for name in (case, f"{case}{DEGRADED_SUFFIX}"))


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

# IPM with every post-solve pass on and the tuned numeric overrides a
# production deployment typically runs with.
_IPM_FULL = PipelineConfig(
    algorithm="ipm",
    bad_data=True,
    v_refine=True,
    v_refine_factor=0.6,
    v_mirror=True,
    v_mirror_min_lift=0.005,
    v_mirror_cross_at=True,
    v_mirror_chain=True,
    refine_merge=True,
    anti_overshoot=False,
    reconcile_respect_bounds=True,
    relax_generation_min_to_injection=True,
    voltage_filter_min_nominal_kv=0.0,
    voltage_filter_deactivate_nonphysical=True,
    flow_sigma_floor_kv_frac=0.01,
    ipm_prior_sigma2_bus_equiv_pu=1.0,
    ipm_boxed_balance_weight_factor=1.0,
)


@dataclass(frozen=True)
class ConfigSpec:
    """A named harness configuration.

    ``warm_start_from`` set: the input is first solved with that configuration
    and the recorded run is the second one, warm-started from the first
    (``init_state``).
    """

    name: str
    description: str
    config: PipelineConfig
    warm_start_from: PipelineConfig | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "description": self.description,
            "config": dataclasses.asdict(self.config),
        }
        if self.warm_start_from is not None:
            out["warm_start_from"] = dataclasses.asdict(self.warm_start_from)
        return out


CONFIGS: dict[str, ConfigSpec] = {
    spec.name: spec
    for spec in (
        ConfigSpec("default", "library defaults (WLS)", PipelineConfig()),
        ConfigSpec(
            "ipm_full",
            "IPM, all post-solve passes, merged refine re-solve",
            _IPM_FULL,
        ),
        ConfigSpec(
            "ipm_sequential",
            "ipm_full with one re-solve per refine pass (refine_merge=False)",
            dataclasses.replace(_IPM_FULL, refine_merge=False),
        ),
        ConfigSpec(
            "anti_overshoot",
            "ipm_full plus anti-overshoot; ceiling lowered to 1.04 so that "
            "IEEE operating points cross it",
            dataclasses.replace(_IPM_FULL, anti_overshoot=True, anti_overshoot_ceiling=1.04),
        ),
        ConfigSpec(
            "shunt_sanity",
            "ipm_full plus shunt sanity trial runs",
            dataclasses.replace(_IPM_FULL, shunt_sanity=True),
        ),
        ConfigSpec(
            "warm_start",
            "ipm_full warm-started from a default (WLS) solve of the same input",
            _IPM_FULL,
            warm_start_from=PipelineConfig(),
        ),
    )
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def degrade(se_input: SEInput) -> SEInput:
    """Deterministic degraded variant of a fully measured input.

    Edits (all derived from the input itself, no randomness):

    1. Box bounds on every ``exist_gen`` / ``exist_load`` node around its
       current generation / load (IPM box variables, boxed balance weight,
       bounds-aware reconcile).
    2. Real |V| readings kept only at generator nodes and at the shunt target
       (below): the other nodes get flat pseudo-V placeholders (chain mirror,
       anti-overshoot candidates).
    3. ``max(1, n_nodes // 30)`` blind nodes - non-generator nodes of lowest
       degree (ties by id) - lose every real reading on the node and on the
       branch ends at the node (v_mirror blind clusters).
    4. The largest |P| branch-flow reading is sign-flipped and the median
       reactive injection reading gets +60 Mvar (bad-data flip / reject /
       damp).
    5. A 100 Mvar reactor at nominal voltage is added to the shunt table at
       the highest-degree non-generator node, which keeps its |V| reading
       (shunt sanity candidate).
    """
    model = se_input.model
    nodes = model.nodes.to_numpy().copy()
    branches = model.branches.to_numpy().copy()
    meas = model.measurements.to_numpy().copy()
    generators = model.generators.to_numpy().copy()

    for i in range(len(nodes)):
        if nodes["exist_gen"][i]:
            pg = float(nodes["generation_p"][i])
            qg = float(nodes["generation_q"][i])
            nodes["generation_p_min"][i] = min(pg, 0.0) - 1.0
            nodes["generation_p_max"][i] = 1.5 * max(pg, 0.0) + 10.0
            nodes["generation_q_min"][i] = -abs(qg) - 50.0
            nodes["generation_q_max"][i] = abs(qg) + 50.0
        if nodes["exist_load"][i]:
            pl = float(nodes["load_p"][i])
            ql = float(nodes["load_q"][i])
            nodes["load_p_min"][i] = pl - 0.2 * abs(pl) - 5.0
            nodes["load_p_max"][i] = pl + 0.2 * abs(pl) + 5.0
            nodes["load_q_min"][i] = ql - 0.3 * abs(ql) - 5.0
            nodes["load_q_max"][i] = ql + 0.3 * abs(ql) + 5.0

    gen_nodes = {int(n) for n, g in zip(nodes["id"], nodes["exist_gen"], strict=True) if g}
    incident: dict[int, list[int]] = {int(n): [] for n in nodes["id"]}
    for k in range(len(branches)):
        incident[int(branches["from_node"][k])].append(k)
        incident[int(branches["to_node"][k])].append(k)
    by_degree = sorted((len(ks), nid) for nid, ks in incident.items() if nid not in gen_nodes)
    n_blind = max(1, len(nodes) // 30)
    blind = [nid for _deg, nid in by_degree[:n_blind]]
    shunt_node = by_degree[-1][1]

    on_node = meas["object_type"] == OBJ_NODE
    on_branch = meas["object_type"] == OBJ_BRANCH
    is_v = on_node & (meas["measurement_type"] == KIND_VOLTAGE)
    keep_v = np.isin(meas["object_id"], [*sorted(gen_nodes), shunt_node])
    keep = ~(is_v & ~keep_v)
    for nid in blind:
        keep &= ~(on_node & (meas["object_id"] == nid))
        for k in incident[nid]:
            side = 0 if int(branches["from_node"][k]) == nid else 1
            keep &= ~(
                on_branch
                & (meas["object_id"] == int(branches["id"][k]))
                & (meas["branch_side"] == side)
            )
    meas = meas[keep]

    is_p_flow = (meas["object_type"] == OBJ_BRANCH) & (meas["measurement_type"] == KIND_POWER_P)
    j_flip = int(np.argmax(np.where(is_p_flow, np.abs(meas["value"]), -1.0)))
    meas["value"][j_flip] = -meas["value"][j_flip]
    q_inj = np.flatnonzero(
        (meas["object_type"] == OBJ_NODE) & (meas["measurement_type"] == KIND_POWER_INJECTION_Q)
    )
    meas["value"][q_inj[len(q_inj) // 2]] += 60.0

    shunts = np.zeros(1, dtype=SE_INPUT.shunts.input_dtype())
    vnom = float(nodes["voltage_nominal"][nodes["id"] == shunt_node][0])
    shunts["id"] = 1
    shunts["node_id"] = shunt_node
    shunts["susceptance"] = -100.0 / vnom**2
    shunts["status"] = True

    working = Working.from_arrays(
        nodes=nodes,
        branches=branches,
        measurements=meas,
        generators=generators,
        shunts=shunts,
    )
    return SEInput(model=working, derived=se_input.derived)


def load_ieee_input(name: str) -> SEInput:
    """Load an IEEE harness input by name (``caseNN`` or ``caseNN_degraded``)."""
    case = name.removesuffix(DEGRADED_SUFFIX)
    if case not in IEEE_CASES:
        raise KeyError(f"unknown IEEE input {name!r}; known: {', '.join(IEEE_INPUTS)}")
    se_input = load_se_input_npz(IEEE_DIR / f"{case}.npz")
    return degrade(se_input) if name.endswith(DEGRADED_SUFFIX) else se_input


def ieee_source(name: str) -> dict[str, Any]:
    case = name.removesuffix(DEGRADED_SUFFIX)
    scenario = "degraded" if name.endswith(DEGRADED_SUFFIX) else "as_is"
    return {"kind": "ieee", "case": case, "scenario": scenario}


def discover_inputs(directory: Path) -> dict[str, Path]:
    """Contract ``.npz`` inputs of a directory, keyed by file stem (sorted)."""
    return {p.stem: p for p in sorted(Path(directory).glob("*.npz"))}


def input_fingerprint(se_input: SEInput) -> str:
    """sha256 over the input tables of the model (structure + bytes)."""
    digest = hashlib.sha256()
    for name in (t.name for t in SE_INPUT.tables()):
        coll = getattr(se_input.model, name, None)
        if coll is None:
            continue
        arr = np.ascontiguousarray(coll.to_numpy())
        digest.update(name.encode())
        digest.update(str(arr.dtype.descr).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

_MODEL_TABLES = (
    "nodes",
    "branches",
    "measurements",
    "generators",
    "tap_steps",
    "load_characteristics",
    "shunts",
)
_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,40}$")
TEXT_MARKER = "<text>"
_DROPPED_EVENT_KEYS = frozenset({"duration_ms", "reason"})


@dataclass
class Snapshot:
    """Result of one (input, config) run: arrays plus JSON-able metadata."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _columns(prefix: str, arr: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in arr.dtype.names or ():
        col = np.ascontiguousarray(arr[name])
        if col.dtype.kind == "O":
            col = col.astype(str)
        out[f"{prefix}.{name}"] = col
    return out


def _records(prefix: str, rows: list[Any]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    for f in dataclasses.fields(rows[0]):
        out[f"{prefix}.{f.name}"] = np.asarray([getattr(r, f.name) for r in rows])
    return out


def normalize(value: Any) -> Any:
    """JSON-able canonical form of an event payload value."""
    if isinstance(value, Mapping):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [normalize(v) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return [normalize(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if value is None:
        return None
    text = str(value)
    return text if _TOKEN.match(text) else TEXT_MARKER


def normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): normalize(v) for k, v in event.items() if k not in _DROPPED_EVENT_KEYS}


def capture(out: SEOutput, events: list[dict[str, Any]]) -> Snapshot:
    """Build a :class:`Snapshot` from a contract run output and its events."""
    arrays: dict[str, np.ndarray] = {}
    for table in ("nodes", "branches", "measurements"):
        arrays.update(_columns(f"output.{table}", getattr(out, table)))
    arrays["output.v_pu"] = np.asarray(out.v_pu)
    arrays["output.delta_rad"] = np.asarray(out.delta_rad)

    result = out.result
    chi2: dict[str, Any] | None = None
    if result is not None:
        for table in _MODEL_TABLES:
            coll = getattr(result.model, table, None)
            if coll is not None:
                arrays.update(_columns(f"model.{table}", coll.to_numpy()))
        arrays.update(_records("quality.worst_residuals", result.worst_residuals))
        arrays.update(_records("quality.worst_imbalance", result.worst_imbalance))
        arrays["quality.observability_warnings"] = np.asarray(
            result.observability_warnings, dtype=np.int64
        )
        if result.chi2 is not None:
            chi2 = normalize(dataclasses.asdict(result.chi2))

    meta = {
        "error": None,
        "scalars": {
            "success": bool(out.success),
            "iterations": int(out.iterations),
            "objective_value": float(out.objective_value),
            "algorithm": str(out.algorithm),
            "convergence_status": str(out.convergence_status),
            "chi2": chi2,
        },
        "events": [normalize_event(e) for e in events],
    }
    return Snapshot(arrays=arrays, meta=meta)


def run_config(se_input: SEInput, spec: ConfigSpec) -> Snapshot:
    """Run one harness configuration on ``se_input`` and capture the snapshot.

    An exception is captured as ``meta["error"]`` (exception type name) so a
    run that starts or stops failing shows up as a difference.
    """
    events: list[dict[str, Any]] = []
    try:
        init_state = None
        if spec.warm_start_from is not None:
            init_state = run_se(se_input, config=spec.warm_start_from)
        out = run_se(se_input, config=spec.config, on_event=events.append, init_state=init_state)
    except Exception as exc:
        return Snapshot(
            meta={
                "error": type(exc).__name__,
                "events": [normalize_event(e) for e in events],
            }
        )
    return capture(out, events)


# ---------------------------------------------------------------------------
# Golden set: in-memory form, record, save, load
# ---------------------------------------------------------------------------


@dataclass
class InputRecord:
    """Snapshots of one input, keyed by configuration name."""

    name: str
    source: dict[str, Any]
    fingerprint: str
    snapshots: dict[str, Snapshot] = field(default_factory=dict)


@dataclass
class GoldenSet:
    environment: dict[str, Any] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, InputRecord] = field(default_factory=dict)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()


def environment() -> dict[str, Any]:
    """Versions and settings that can influence numeric results."""
    import os

    from gridstate import __version__
    from gridstate.algorithms.kkt_solver import resolve_backend

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "gridstate": __version__,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": _dist_version("scipy"),
        "cvxopt": _dist_version("cvxopt"),
        "kkt_backend": resolve_backend("auto"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def record_input(
    name: str,
    se_input: SEInput,
    source: dict[str, Any],
    configs: Iterable[str],
) -> InputRecord:
    rec = InputRecord(name=name, source=source, fingerprint=input_fingerprint(se_input))
    for cfg_name in configs:
        rec.snapshots[cfg_name] = run_config(se_input, CONFIGS[cfg_name])
    return rec


def record(
    *,
    ieee: Iterable[str] = IEEE_INPUTS,
    files: Mapping[str, Path] | None = None,
    configs: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> GoldenSet:
    """Run the harness matrix and return the in-memory golden set."""
    cfg_names = list(configs) if configs is not None else list(CONFIGS)
    unknown = [c for c in cfg_names if c not in CONFIGS]
    if unknown:
        raise KeyError(f"unknown config(s) {unknown}; known: {', '.join(CONFIGS)}")
    gset = GoldenSet(
        environment=environment(),
        configs={c: CONFIGS[c].to_json() for c in cfg_names},
    )
    for name in ieee:
        if progress is not None:
            progress(name)
        gset.inputs[name] = record_input(name, load_ieee_input(name), ieee_source(name), cfg_names)
    for name, path in (files or {}).items():
        if name in gset.inputs:
            raise ValueError(f"input name {name!r} collides with an IEEE input")
        if progress is not None:
            progress(name)
        source = {"kind": "file", "file": Path(path).name}
        gset.inputs[name] = record_input(name, load_se_input_npz(path), source, cfg_names)
    return gset


def rerun(
    reference: GoldenSet,
    *,
    files: Mapping[str, Path] | None = None,
    only: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> GoldenSet:
    """Re-run every (input, config) present in ``reference`` with the current code.

    IEEE inputs are rebuilt from the committed fixtures; file inputs are looked
    up by name in ``files`` (missing ones are left out and reported by
    :func:`compare`).
    """
    files = files or {}
    wanted = set(only) if only is not None else None
    gset = GoldenSet(
        environment=environment(),
        configs={c: CONFIGS[c].to_json() for c in reference.configs if c in CONFIGS},
    )
    for name, ref in reference.inputs.items():
        if wanted is not None and name not in wanted:
            continue
        cfg_names = [c for c in ref.snapshots if c in CONFIGS]
        if ref.source.get("kind") == "ieee":
            se_input = load_ieee_input(name)
        elif name in files:
            se_input = load_se_input_npz(files[name])
        else:
            continue
        if progress is not None:
            progress(name)
        gset.inputs[name] = record_input(name, se_input, ref.source, cfg_names)
    return gset


def _pack(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Group 1-D columns sharing a prefix and a length into one structured array.

    Purely a storage optimization (one archive member per table instead of one
    per column); :func:`_unpack` restores the flat column keys.
    """
    groups: dict[str, list[str]] = {}
    for key in arrays:
        groups.setdefault(key.rpartition(".")[0], []).append(key)
    out: dict[str, np.ndarray] = {}
    for prefix, keys in groups.items():
        cols = [arrays[k] for k in keys]
        if prefix and all(c.ndim == 1 and len(c) == len(cols[0]) for c in cols):
            names = [k.rpartition(".")[2] for k in keys]
            packed = np.empty(
                len(cols[0]), dtype=[(n, c.dtype) for n, c in zip(names, cols, strict=True)]
            )
            for n, c in zip(names, cols, strict=True):
                packed[n] = c
            out[prefix] = packed
        else:
            out.update({k: arrays[k] for k in keys})
    return out


def _unpack(key: str, arr: np.ndarray) -> dict[str, np.ndarray]:
    if arr.dtype.names is None:
        return {key: arr}
    return {f"{key}.{n}": np.ascontiguousarray(arr[n]) for n in arr.dtype.names}


def _dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def save(gset: GoldenSet, directory: Path) -> None:
    """Write a golden set: ``manifest.json`` + ``<input>.npz`` / ``<input>.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT_VERSION,
        "environment": gset.environment,
        "configs": gset.configs,
        "inputs": {
            name: {
                "source": rec.source,
                "fingerprint": rec.fingerprint,
                "configs": list(rec.snapshots),
            }
            for name, rec in gset.inputs.items()
        },
    }
    for name, rec in gset.inputs.items():
        arrays = {
            f"{cfg}/{key}": arr
            for cfg, snap in rec.snapshots.items()
            for key, arr in _pack(snap.arrays).items()
        }
        np.savez_compressed(directory / f"{name}.npz", **arrays)  # type: ignore[arg-type]
        _dump_json(directory / f"{name}.json", {cfg: s.meta for cfg, s in rec.snapshots.items()})
    _dump_json(directory / MANIFEST_NAME, manifest)


def load(directory: Path) -> GoldenSet:
    """Read a golden set written by :func:`save`."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{directory}: golden format {manifest.get('format')!r}, expected {FORMAT_VERSION}"
        )
    gset = GoldenSet(environment=manifest["environment"], configs=manifest["configs"])
    for name, entry in manifest["inputs"].items():
        metas = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        rec = InputRecord(name=name, source=entry["source"], fingerprint=entry["fingerprint"])
        for cfg in entry["configs"]:
            rec.snapshots[cfg] = Snapshot(meta=metas[cfg])
        with np.load(directory / f"{name}.npz", allow_pickle=False) as npz:
            for key in npz.files:
                cfg, _, arr_key = key.partition("/")
                rec.snapshots[cfg].arrays.update(_unpack(arr_key, np.asarray(npz[key])))
        gset.inputs[name] = rec
    return gset


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """Comparison mode: ``strict`` (bit-exact) or tolerant with ``rtol``/``atol``."""

    strict: bool = False
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    def describe(self) -> str:
        if self.strict:
            return "strict (bit-exact)"
        return f"tolerant (rtol={self.rtol:g}, atol={self.atol:g})"

    def scaled(self, factor: float) -> Tolerance:
        if self.strict:
            return self
        return dataclasses.replace(self, rtol=self.rtol * factor, atol=self.atol * factor)


@dataclass(frozen=True)
class Difference:
    input: str
    config: str
    item: str
    detail: str

    def format(self) -> str:
        return f"{self.input} / {self.config}: {self.item}: {self.detail}"


@dataclass
class CompareReport:
    tolerance: Tolerance
    differences: list[Difference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    compared: int = 0

    @property
    def ok(self) -> bool:
        return not self.differences

    def format(self, *, max_items: int = 200) -> str:
        lines = [f"golden compare: {self.compared} snapshot(s), mode {self.tolerance.describe()}"]
        lines += [f"  note: {n}" for n in self.notes]
        for d in self.differences[:max_items]:
            lines.append(f"  DIFF {d.format()}")
        if len(self.differences) > max_items:
            lines.append(f"  ... {len(self.differences) - max_items} more")
        verdict = "identical" if self.ok else f"{len(self.differences)} difference(s)"
        lines.append(f"result: {verdict}")
        return "\n".join(lines)


def _fmt(x: Any) -> str:
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.17g}"
    return repr(x.item() if isinstance(x, np.generic) else x)


def compare_arrays(ref: np.ndarray, cand: np.ndarray, tol: Tolerance) -> str | None:
    """``None`` if equal under ``tol``, else a one-line description of the diff."""
    if ref.shape != cand.shape:
        return f"shape {ref.shape} != {cand.shape}"
    if tol.strict and ref.dtype != cand.dtype:
        return f"dtype {ref.dtype} != {cand.dtype}"
    float_kinds = "fc"
    if ref.dtype.kind in float_kinds or cand.dtype.kind in float_kinds:
        if ref.dtype.kind not in "biufc" or cand.dtype.kind not in "biufc":
            return f"dtype {ref.dtype} != {cand.dtype}"
        a = ref.astype(np.complex128 if "c" in (ref.dtype.kind, cand.dtype.kind) else np.float64)
        b = cand.astype(a.dtype)
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        if tol.strict:
            bad = ~((a == b) | (nan_a & nan_b))
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                close = np.abs(a - b) <= tol.atol + tol.rtol * np.abs(a)
            bad = ~((a == b) | (nan_a & nan_b) | (close & ~nan_a & ~nan_b))
        if not bad.any():
            return None
        n_bad = int(bad.sum())
        both = bad & ~nan_a & ~nan_b & np.isfinite(a) & np.isfinite(b)
        parts = [f"{n_bad}/{a.size} differ"]
        if both.any():
            absd = np.abs(a - b)[both]
            reld = absd / np.maximum(np.abs(a[both]), np.finfo(np.float64).tiny)
            parts.append(f"max_abs {float(absd.max()):.3e}, max_rel {float(reld.max()):.3e}")
        n_nan = int((nan_a != nan_b).sum())
        if n_nan:
            parts.append(f"NaN positions differ at {n_nan}")
        first = np.unravel_index(int(np.flatnonzero(bad)[0]), bad.shape)
        parts.append(
            f"first at {tuple(int(i) for i in first)}: {_fmt(a[first])} -> {_fmt(b[first])}"
        )
        return "; ".join(parts)
    if ref.dtype.kind != cand.dtype.kind and not (
        ref.dtype.kind in "biu" and cand.dtype.kind in "biu"
    ):
        return f"dtype {ref.dtype} != {cand.dtype}"
    bad = ref != cand
    if not np.any(bad):
        return None
    bad = np.asarray(bad)
    first = np.unravel_index(int(np.flatnonzero(bad)[0]), bad.shape)
    return (
        f"{int(bad.sum())}/{ref.size} differ; first at {tuple(int(i) for i in first)}: "
        f"{_fmt(ref[first])} -> {_fmt(cand[first])}"
    )


def _values_equal(a: Any, b: Any, tol: Tolerance) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b or (a == b and type(a) is type(b))
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        if tol.strict:
            return fa == fb and type(a) is type(b)
        return fa == fb or abs(fa - fb) <= tol.atol + tol.rtol * abs(fa)
    return bool(a == b)


def compare_values(ref: Any, cand: Any, tol: Tolerance, path: str = "") -> list[str]:
    """Recursive diff of JSON-like values; returns ``path: ref -> cand`` lines."""
    if isinstance(ref, dict) and isinstance(cand, dict):
        out: list[str] = []
        for key in ref.keys() | cand.keys():
            sub = f"{path}.{key}" if path else str(key)
            if key not in cand:
                out.append(f"{sub}: missing in candidate")
            elif key not in ref:
                out.append(f"{sub}: not in reference")
            else:
                out.extend(compare_values(ref[key], cand[key], tol, sub))
        return sorted(out)
    if isinstance(ref, list) and isinstance(cand, list):
        if len(ref) != len(cand):
            return [f"{path}: length {len(ref)} != {len(cand)}"]
        out = []
        for i, (r, c) in enumerate(zip(ref, cand, strict=True)):
            out.extend(compare_values(r, c, tol, f"{path}[{i}]"))
        return out
    if _values_equal(ref, cand, tol):
        return []
    return [f"{path}: {_fmt(ref)} -> {_fmt(cand)}"]


def _event_label(e: Mapping[str, Any]) -> str:
    return f"{e.get('type')}:{e.get('name', '')}"


def compare_events(
    ref: list[dict[str, Any]], cand: list[dict[str, Any]], tol: Tolerance
) -> list[str]:
    """Event sequence must match exactly (type, name); payloads under ``tol``."""
    import difflib

    ref_labels = [_event_label(e) for e in ref]
    cand_labels = [_event_label(e) for e in cand]
    if ref_labels != cand_labels:
        diff = difflib.unified_diff(
            ref_labels, cand_labels, "reference", "candidate", n=1, lineterm=""
        )
        return ["event sequence differs: " + " | ".join(list(diff)[2:])]
    out: list[str] = []
    for i, (r, c) in enumerate(zip(ref, cand, strict=True)):
        out.extend(compare_values(r, c, tol, f"#{i} {_event_label(r)}"))
    return out


def compare_snapshots(
    ref: Snapshot, cand: Snapshot, tol: Tolerance, *, input_name: str, config: str
) -> list[Difference]:
    diffs: list[Difference] = []

    def add(item: str, detail: str) -> None:
        diffs.append(Difference(input_name, config, item, detail))

    if ref.meta.get("error") != cand.meta.get("error"):
        add("error", f"{ref.meta.get('error')!r} -> {cand.meta.get('error')!r}")
    for line in compare_values(ref.meta.get("scalars"), cand.meta.get("scalars"), tol):
        add("scalar", line)
    for line in compare_events(ref.meta.get("events", []), cand.meta.get("events", []), tol):
        add("events", line)
    for key in sorted(ref.arrays.keys() | cand.arrays.keys()):
        if key not in cand.arrays:
            add(key, "missing in candidate")
        elif key not in ref.arrays:
            add(key, "not in reference")
        else:
            detail = compare_arrays(ref.arrays[key], cand.arrays[key], tol)
            if detail is not None:
                add(key, detail)
    return diffs


def compare(reference: GoldenSet, candidate: GoldenSet, tol: Tolerance) -> CompareReport:
    """Compare every (input, config) snapshot of ``reference`` with ``candidate``."""
    report = CompareReport(tolerance=tol)
    for name, ref in reference.inputs.items():
        cand = candidate.inputs.get(name)
        if cand is None:
            report.differences.append(Difference(name, "*", "input", "missing in candidate"))
            continue
        if ref.fingerprint != cand.fingerprint:
            report.notes.append(f"{name}: input fingerprint changed (fixture or scenario edited)")
        for cfg, ref_snap in ref.snapshots.items():
            cand_snap = cand.snapshots.get(cfg)
            if cand_snap is None:
                report.differences.append(Difference(name, cfg, "config", "missing in candidate"))
                continue
            report.compared += 1
            entry_tol = tol
            widen = TOLERANCE_SCALE.get((name, cfg))
            if widen is not None and not tol.strict:
                entry_tol = tol.scaled(widen[0])
                report.notes.append(f"{name} / {cfg}: tolerance x{widen[0]:g} ({widen[1]})")
            report.differences.extend(
                compare_snapshots(ref_snap, cand_snap, entry_tol, input_name=name, config=cfg)
            )
    for name in candidate.inputs.keys() - reference.inputs.keys():
        report.notes.append(f"{name}: not in reference (ignored)")
    for cfg, spec in reference.configs.items():
        cur = candidate.configs.get(cfg)
        if cur is not None and cur != spec:
            changed = compare_values(spec, cur, Tolerance(strict=True))
            report.notes.append(f"config {cfg!r} definition changed: {'; '.join(changed[:5])}")
    env_ref, env_cand = reference.environment, candidate.environment
    for key in ("gridstate", "git_sha", "numpy", "scipy", "cvxopt", "kkt_backend", "python"):
        if env_ref.get(key) != env_cand.get(key):
            report.notes.append(f"environment {key}: {env_ref.get(key)} -> {env_cand.get(key)}")
    return report
