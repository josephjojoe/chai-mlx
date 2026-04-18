from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn

from chai_mlx.model.confidence import ConfidenceHead
from chai_mlx.config import (
    AtomBlockConfig,
    ChaiConfig,
    ConfidenceConfig,
    DiffusionConfig,
    FeatureDims,
    HiddenDims,
    MSAConfig,
    PairformerConfig,
    TemplateConfig,
)
from chai_mlx.model.diffusion import DiffusionModule
from chai_mlx.model.embeddings import InputEmbedder
from chai_mlx.data.featurize import featurize as _featurize
from chai_mlx.model.ranking import Ranker
from chai_mlx.model.trunk import Trunk
from chai_mlx.data.types import (
    ConfidenceOutputs,
    DiffusionCache,
    EmbeddingOutputs,
    FeatureContext,
    InputBundle,
    RankingOutputs,
    TrunkOutputs,
)
from chai_mlx.io.weights.load import load_safetensors
from chai_mlx.nn.layers.common import FP32LayerNorm
from chai_mlx.utils import resolve_dtype


def _preserve_fp32_param_keys(model: nn.Module) -> set[str]:
    """Parameter keys that should remain float32 in mixed-precision mode.

    TorchScript keeps affine parameters for FP32 layer norms in float32 and
    also uses float32 ``query_bias`` / ``out_scalers`` parameters without an
    explicit cast to bf16 before applying them.
    """
    keep: set[str] = set()
    for path, module in model.named_modules():
        prefix = f"{path}." if path else ""
        if isinstance(module, FP32LayerNorm):
            for name in ("weight", "bias"):
                if hasattr(module, name):
                    keep.add(f"{prefix}{name}")
        for name in ("query_bias", "out_scalers"):
            if hasattr(module, name):
                keep.add(f"{prefix}{name}")
    return keep


def _cast_weights(model: nn.Module, dtype: mx.Dtype) -> None:
    """Cast all model parameters to *dtype* in-place."""
    from mlx.utils import tree_flatten

    preserve_fp32 = _preserve_fp32_param_keys(model)
    pairs = []
    for k, v in tree_flatten(model.parameters()):
        if k in preserve_fp32:
            continue
        if isinstance(v, mx.array) and v.dtype != dtype:
            pairs.append((k, v.astype(dtype)))
    if pairs:
        model.load_weights(pairs, strict=False)


def load_pretrained_config(
    path_or_repo: str | Path,
    *,
    compute_dtype: str | None = None,
) -> tuple[Path, ChaiConfig]:
    """Resolve a pretrained model path and load its config."""
    from dataclasses import fields as dc_fields

    path = Path(path_or_repo)
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path_or_repo)))
        except ImportError:
            raise ValueError(
                f"{path_or_repo} is not a local directory and "
                "huggingface_hub is not installed for remote download"
            )

    config_path = path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            raw = json.load(f)
        _NESTED = {
            "feature_dims": FeatureDims,
            "hidden": HiddenDims,
            "atom_blocks": AtomBlockConfig,
            "pairformer": PairformerConfig,
            "diffusion": DiffusionConfig,
            "templates": TemplateConfig,
            "msa": MSAConfig,
            "confidence": ConfidenceConfig,
        }
        if "supported_token_sizes" in raw and isinstance(raw["supported_token_sizes"], list):
            raw["supported_token_sizes"] = tuple(raw["supported_token_sizes"])
        for key, cls_ in _NESTED.items():
            if key in raw and isinstance(raw[key], dict):
                v = raw[key]
                if "distance_bin_edges" in v and isinstance(v["distance_bin_edges"], list):
                    v["distance_bin_edges"] = tuple(v["distance_bin_edges"])
                if "supported_token_sizes" in v and isinstance(v["supported_token_sizes"], list):
                    v["supported_token_sizes"] = tuple(v["supported_token_sizes"])
                raw[key] = cls_(**v)
        cfg = ChaiConfig(**raw)
    else:
        cfg = ChaiConfig()

    if compute_dtype is not None:
        cfg_dict = {f.name: getattr(cfg, f.name) for f in dc_fields(cfg)}
        cfg_dict["compute_dtype"] = compute_dtype
        cfg = ChaiConfig(**cfg_dict)

    return path, cfg


# Periodically drop Metal's allocator cache during long diffusion roll-outs.
# Without this MLX's pool can grow unboundedly across hundreds of steps.
# Calling ``mx.clear_cache`` on every step forces a sync that hurts
# throughput; 16 is the empirical sweet spot that bounds peak memory
# without noticeably disrupting the compute pipeline.
_DIFFUSION_CACHE_CLEAR_INTERVAL = 16


@dataclass
class FoldOutputs:
    """Debug fold outputs with full intermediate tensors."""
    context: FeatureContext
    embeddings: EmbeddingOutputs
    trunk: TrunkOutputs
    coords: mx.array
    confidence: ConfidenceOutputs
    ranking: RankingOutputs


@dataclass
class InferenceOutputs:
    """Production fold outputs (final tensors only)."""
    coords: mx.array
    confidence: ConfidenceOutputs
    ranking: RankingOutputs


class ChaiMLX(nn.Module):
    def __init__(self, cfg: ChaiConfig | None = None) -> None:
        super().__init__()
        self.cfg = ChaiConfig() if cfg is None else cfg
        self.input_embedder = InputEmbedder(self.cfg)
        self.trunk_module = Trunk(self.cfg)
        self.diffusion_module = DiffusionModule(self.cfg)
        self.confidence_head = ConfidenceHead(self.cfg)
        self.ranker = Ranker(self.cfg)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        strict: bool = True,
        compute_dtype: str | None = None,
    ) -> "ChaiMLX":
        """Load a pretrained model from a local directory or HuggingFace repo.

        The directory should contain ``config.json`` and either
        ``model.safetensors`` or sharded safetensors with an index file.

        Parameters
        ----------
        compute_dtype : str, optional
            Override the config's ``compute_dtype``.  Pass ``"float32"`` to
            disable mixed precision, or ``"bfloat16"`` (the default in
            :class:`ChaiConfig`) for half-precision inference.
        """
        path, cfg = load_pretrained_config(path_or_repo, compute_dtype=compute_dtype)

        model = cls(cfg)
        load_safetensors(model, path, strict=strict)

        dtype = resolve_dtype(cfg)
        if dtype != mx.float32:
            _cast_weights(model, dtype)

        return model

    def featurize(self, inputs: FeatureContext | InputBundle | dict) -> FeatureContext:
        return _featurize(inputs)

    def embed_inputs(self, ctx: FeatureContext) -> EmbeddingOutputs:
        return self.input_embedder(ctx)

    def trunk(self, emb: EmbeddingOutputs, *, recycles: int = 3) -> TrunkOutputs:
        return self.trunk_module(emb, recycles=recycles)

    def prepare_diffusion_cache(
        self,
        trunk_out: TrunkOutputs,
    ) -> DiffusionCache:
        return self.diffusion_module.prepare_cache(trunk_out)

    def schedule(self, num_steps: int | None = None) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        return self.diffusion_module.schedule(num_steps=num_steps)

    def init_noise(self, batch_size: int, num_samples: int, structure) -> mx.array:
        return self.diffusion_module.init_noise(batch_size, num_samples, structure)

    def denoise(
        self,
        cache: DiffusionCache,
        coords: mx.array,
        sigma: mx.array,
    ) -> mx.array:
        return self.diffusion_module.denoise(cache, coords, sigma)

    def diffusion_step(
        self,
        cache: DiffusionCache,
        coords: mx.array,
        sigma_curr: mx.array | float,
        sigma_next: mx.array | float,
        gamma: mx.array | float,
    ) -> mx.array:
        return self.diffusion_module.diffusion_step(
            cache,
            coords,
            sigma_curr,
            sigma_next,
            gamma,
        )

    def confidence(self, trunk_out: TrunkOutputs, coords: mx.array) -> ConfidenceOutputs:
        return self.confidence_head(trunk_out, coords)

    def rank_outputs(
        self,
        conf: ConfidenceOutputs,
        coords: mx.array,
        structure=None,
    ) -> RankingOutputs:
        structure = conf.structure_inputs if structure is None else structure
        if structure is None:
            raise ValueError("structure must be provided when confidence outputs do not carry it")
        return self.ranker(conf, coords, structure)

    @staticmethod
    def _without_raw_features(ctx: FeatureContext) -> FeatureContext:
        """Return a FeatureContext view that drops heavy raw feature tensors.

        Keeping ``raw_features`` alive after ``embed_inputs`` can retain a large
        amount of memory (especially token-pair/template feature blocks).  This
        helper preserves all fields needed by later stages while clearing only
        the no-longer-needed raw feature dict.
        """
        if ctx.raw_features is None:
            return ctx
        return FeatureContext(
            token_features=ctx.token_features,
            token_pair_features=ctx.token_pair_features,
            atom_features=ctx.atom_features,
            atom_pair_features=ctx.atom_pair_features,
            msa_features=ctx.msa_features,
            template_features=ctx.template_features,
            structure_inputs=ctx.structure_inputs,
            bond_adjacency=ctx.bond_adjacency,
            raw_features=None,
        )

    def run_inference(
        self,
        inputs: FeatureContext | InputBundle | dict,
        *,
        recycles: int = 3,
        num_samples: int = 5,
        num_steps: int | None = None,
    ) -> InferenceOutputs:
        """Run production inference (no intermediate retention)."""
        ctx = self.featurize(inputs)
        emb = self.embed_inputs(ctx)
        # Raw features are only needed during embedding.
        ctx = self._without_raw_features(ctx)
        structure = ctx.structure_inputs
        batch_size = emb.token_single_input.shape[0]
        del ctx
        mx.clear_cache()

        trunk_out = self.trunk(emb, recycles=recycles)
        cache = self.prepare_diffusion_cache(trunk_out)
        mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
                cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)
        del emb
        mx.clear_cache()

        coords = self.init_noise(batch_size, num_samples, structure)
        for step_idx, (sigma_curr, sigma_next, gamma) in enumerate(self.schedule(num_steps=num_steps), start=1):
            coords = self.diffusion_step(
                cache,
                coords,
                sigma_curr,
                sigma_next,
                gamma,
            )
            mx.eval(coords)
            if step_idx % _DIFFUSION_CACHE_CLEAR_INTERVAL == 0:
                mx.clear_cache()
        mx.clear_cache()

        conf_full = self.confidence(trunk_out, coords)
        rank = self.ranker(conf_full, coords, structure)
        conf = ConfidenceOutputs(
            pae_logits=conf_full.pae_logits,
            pde_logits=conf_full.pde_logits,
            plddt_logits=conf_full.plddt_logits,
        )
        del conf_full, trunk_out, cache
        mx.clear_cache()
        return InferenceOutputs(
            coords=coords,
            confidence=conf,
            ranking=rank,
        )

    def run_inference_debug(
        self,
        inputs: FeatureContext | InputBundle | dict,
        *,
        recycles: int = 3,
        num_samples: int = 5,
        num_steps: int | None = None,
    ) -> FoldOutputs:
        """Run debug inference and return full intermediate tensors."""
        ctx = self.featurize(inputs)
        emb = self.embed_inputs(ctx)
        trunk_out = self.trunk(emb, recycles=recycles)
        cache = self.prepare_diffusion_cache(trunk_out)
        mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
                cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)
        coords = self.init_noise(emb.token_single_input.shape[0], num_samples, emb.structure_inputs)
        for sigma_curr, sigma_next, gamma in self.schedule(num_steps=num_steps):
            coords = self.diffusion_step(
                cache,
                coords,
                sigma_curr,
                sigma_next,
                gamma,
            )
            mx.eval(coords)
        conf = self.confidence(trunk_out, coords)
        rank = self.ranker(conf, coords, emb.structure_inputs)
        return FoldOutputs(
            context=ctx,
            embeddings=emb,
            trunk=trunk_out,
            coords=coords,
            confidence=conf,
            ranking=rank,
        )
