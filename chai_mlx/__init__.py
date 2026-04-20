from importlib.metadata import PackageNotFoundError, version as _pkg_version

from chai_mlx.config import ChaiConfig
from chai_mlx.data import (
    ConfidenceOutputs,
    DiffusionCache,
    EmbeddingOutputs,
    FeatureContext,
    InputBundle,
    RankingOutputs,
    StructureInputs,
    TrunkOutputs,
    featurize,
    featurize_fasta,
)
from chai_mlx.model import ChaiMLX, FoldOutputs, InferenceOutputs


try:
    __version__ = _pkg_version("chai-mlx")
except PackageNotFoundError:  # pragma: no cover - source-tree fallback
    # Mirrors the literal in pyproject.toml::[project].version so that
    # ``import chai_mlx; chai_mlx.__version__`` keeps working when the
    # package is imported out of a clone that has not been installed
    # (e.g. CI running straight from a checkout with PYTHONPATH).
    __version__ = "0.1.0"


__all__ = [
    "ChaiMLX",
    "ChaiConfig",
    "FoldOutputs",
    "InferenceOutputs",
    "featurize",
    "featurize_fasta",
    "ConfidenceOutputs",
    "DiffusionCache",
    "EmbeddingOutputs",
    "FeatureContext",
    "InputBundle",
    "RankingOutputs",
    "StructureInputs",
    "TrunkOutputs",
    "__version__",
]
