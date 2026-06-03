"""gridstate — power system State Estimation.

The public API is gathered here; see the README for an overview and the
individual function docstrings for usage.
"""

from importlib.metadata import PackageNotFoundError, version

from gridstate.api import estimate
from gridstate.contract import (
    SEInput,
    SEOutput,
    load_se_input,
)
from gridstate.contract import (
    run as run_se,
)
from gridstate.contract.serialize import (
    load_se_input_npz,
    save_se_input,
)
from gridstate.pipeline import (
    PipelineConfig,
)
from gridstate.pipeline import (
    manifest as pipeline_manifest,
)
from gridstate.pipeline import (
    run as run_pipeline,
)
from gridstate.result import (
    Chi2Summary,
    ImbalanceRow,
    ResidualRow,
    SEResult,
)
from gridstate.validation.bad_data import (
    BadDataResult,
    NormalizedResidualReport,
    compute_normalized_residuals_report,
    remove_bad_data,
)
from gridstate.validation.chi2_test import Chi2Result, chi2_analysis
from gridstate.validation.observability import (
    ObservabilityReport,
    analyze_observability,
)


try:
    # Version is derived from git tags by setuptools_scm and baked into the
    # package metadata at build/install time; read it back from there at
    # runtime. The fallback only triggers in a bare source tree that was never
    # installed or built.
    __version__ = version("gridstate")
except PackageNotFoundError:  # pragma: no cover - un-installed source tree
    __version__ = "0+unknown"


__all__ = [
    "BadDataResult",
    "Chi2Result",
    "Chi2Summary",
    "ImbalanceRow",
    "NormalizedResidualReport",
    "ObservabilityReport",
    "PipelineConfig",
    "ResidualRow",
    "SEInput",
    "SEOutput",
    "SEResult",
    "__version__",
    "analyze_observability",
    "chi2_analysis",
    "compute_normalized_residuals_report",
    "estimate",
    "load_se_input",
    "load_se_input_npz",
    "pipeline_manifest",
    "remove_bad_data",
    "run_pipeline",
    "run_se",
    "save_se_input",
]
