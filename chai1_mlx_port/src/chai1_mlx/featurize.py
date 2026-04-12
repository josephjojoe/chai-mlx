from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .types import FeatureContext, InputBundle, StructureInputs


_REQUIRED_KEYS = {
    "token_features",
    "token_pair_features",
    "atom_features",
    "atom_pair_features",
    "msa_features",
    "template_features",
    "structure_inputs",
}


def _coerce_structure_inputs(obj: Any) -> StructureInputs:
    if isinstance(obj, StructureInputs):
        return obj
    if is_dataclass(obj):
        return StructureInputs(**asdict(obj))
    if isinstance(obj, dict):
        return StructureInputs(**obj)
    raise TypeError("structure_inputs must be a StructureInputs instance or dict")


def featurize(inputs: FeatureContext | InputBundle | dict[str, Any]) -> FeatureContext:
    """Return a frontend-independent FeatureContext.

    This port assumes the caller supplies the *final encoded tensors* expected by the
    feature embedding stage. Raw FASTA/MSA/template parsing is deliberately kept out
    of the MLX neural core.
    """

    if isinstance(inputs, FeatureContext):
        return inputs

    if isinstance(inputs, InputBundle):
        if inputs.features is not None:
            return inputs.features
        inputs = inputs.raw

    if not isinstance(inputs, dict):
        raise TypeError(
            "featurize() expects a FeatureContext, InputBundle, or dict of precomputed tensors"
        )

    missing = sorted(_REQUIRED_KEYS - set(inputs))
    if missing:
        raise ValueError(
            "This MLX port expects precomputed encoded feature tensors. Missing keys: "
            + ", ".join(missing)
        )

    payload = dict(inputs)
    payload["structure_inputs"] = _coerce_structure_inputs(payload["structure_inputs"])
    return FeatureContext(**payload)
