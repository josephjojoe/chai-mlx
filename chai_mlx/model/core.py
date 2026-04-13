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
from chai_mlx.utils import resolve_dtype


def _upgrade_linears(module: nn.Module) -> None:
    """Replace every ``nn.Linear`` with :class:`BF16Linear` in-place.

    Walks the module tree and swaps ``__class__`` so that each Linear
    casts its input to bfloat16 before the matmul, matching the
    TorchScript ``torch.to(x, 15)`` pattern.  Weights stay fp32.
    """
    from chai_mlx.nn.layers.common import BF16Linear

    for key, child in module.children().items():
        if isinstance(child, nn.Linear) and type(child) is nn.Linear:
            child.__class__ = BF16Linear
        elif isinstance(child, nn.Module):
            _upgrade_linears(child)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, nn.Linear) and type(item) is nn.Linear:
                    item.__class__ = BF16Linear
                elif isinstance(item, nn.Module):
                    _upgrade_linears(item)


@dataclass
class FoldOutputs:
    context: FeatureContext
    embeddings: EmbeddingOutputs
    trunk: TrunkOutputs
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

        model = cls(cfg)
        load_safetensors(model, path, strict=strict)

        dtype = resolve_dtype(cfg)
        if dtype != mx.float32:
            _upgrade_linears(model)

        return model

    def featurize(self, inputs: FeatureContext | InputBundle | dict) -> FeatureContext:
        return _featurize(inputs)

    def embed_inputs(self, ctx: FeatureContext, *, use_kernel: bool = False) -> EmbeddingOutputs:
        return self.input_embedder(ctx, use_kernel=use_kernel)

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
        *,
        use_kernel: bool = False,
    ) -> mx.array:
        return self.diffusion_module.denoise(
            cache, coords, sigma, use_kernel=use_kernel
        )

    def diffusion_step(
        self,
        cache: DiffusionCache,
        coords: mx.array,
        sigma_curr: mx.array | float,
        sigma_next: mx.array | float,
        gamma: mx.array | float,
        *,
        use_kernel: bool = False,
    ) -> mx.array:
        return self.diffusion_module.diffusion_step(
            cache,
            coords,
            sigma_curr,
            sigma_next,
            gamma,
            use_kernel=use_kernel,
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

    def fold(
        self,
        inputs: FeatureContext | InputBundle | dict,
        *,
        recycles: int = 3,
        num_samples: int = 5,
        num_steps: int | None = None,
        use_kernel: bool = False,
    ) -> FoldOutputs:
        ctx = self.featurize(inputs)
        emb = self.embed_inputs(ctx, use_kernel=use_kernel)
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
                use_kernel=use_kernel,
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
