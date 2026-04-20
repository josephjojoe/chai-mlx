from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class FeatureDims:
    token: int = 2638
    token_pair: int = 163
    atom: int = 395
    atom_pair: int = 14
    msa: int = 42
    templates: int = 76
    bond: int = 1


@dataclass(frozen=True)
class HiddenDims:
    token_single: int = 384
    token_pair: int = 256
    msa: int = 64
    template_pair: int = 64
    atom_single: int = 128
    atom_pair: int = 16
    diffusion: int = 768


@dataclass(frozen=True)
class AtomBlockConfig:
    query_block: int = 32
    kv_block: int = 128
    atom_multiplier: int = 23
    local_blocks: int = 3
    num_heads: int = 4
    head_dim: int = 32


@dataclass(frozen=True)
class PairformerConfig:
    num_blocks: int = 48
    single_heads: int = 16
    single_head_dim: int = 24
    triangle_heads: int = 4
    triangle_head_dim: int = 64
    # NOTE: Transition expansion is hard-coded to 2 in PairformerBlock,
    # matching the TorchScript weight shapes.  These fields are retained
    # for documentation but are not consumed by any module.
    single_transition_mult: int = 2
    pair_transition_mult: int = 2


@dataclass(frozen=True)
class DiffusionConfig:
    num_blocks: int = 16
    num_heads: int = 16
    head_dim: int = 48
    single_cond_dim: int = 384
    working_dim: int = 768
    num_steps: int = 200
    sigma_data: float = 16.0
    s_min: float = 4e-4
    s_max: float = 80.0
    p: float = 7.0
    s_churn: float = 80.0
    s_tmin: float = 4e-4
    s_tmax: float = 80.0
    s_noise: float = 1.003
    second_order: bool = True


@dataclass(frozen=True)
class TemplateConfig:
    max_templates: int = 4
    num_blocks: int = 2
    triangle_heads: int = 4
    triangle_head_dim: int = 32


@dataclass(frozen=True)
class MSAConfig:
    num_outer_product_mean: int = 4
    num_pair_weighted_avg: int = 3
    num_msa_transition: int = 3
    num_pair_transition: int = 4
    num_tri_mult: int = 4
    num_tri_attn: int = 4
    pair_weight_heads: int = 8
    pair_weight_value_dim: int = 32


@dataclass(frozen=True)
class ConfidenceConfig:
    num_blocks: int = 4
    triangle_heads: int = 4
    triangle_head_dim: int = 64
    plddt_bins: int = 50
    plddt_atom_positions: int = 37
    pair_bins: int = 64
    distance_bin_edges: Sequence[float] = (
        3.375,
        4.660714285714286,
        5.946428571428571,
        7.232142857142857,
        8.517857142857142,
        9.803571428571429,
        11.089285714285714,
        12.375,
        13.660714285714286,
        14.946428571428571,
        16.232142857142858,
        17.517857142857142,
        18.803571428571427,
        20.089285714285715,
        21.375,
    )


@dataclass(frozen=True)
class ChaiConfig:
    # Versions the on-disk ``config.json`` schema. Bump when adding
    # required fields that older checkpoints cannot supply. Older
    # checkpoints without this field are assumed to be ``"1"`` so
    # ``from_pretrained`` stays backwards-compatible with every
    # chai-mlx weight release to date; any future bump should print
    # a diagnostic via ``load_pretrained_config``.
    #
    # Kept as a public (non-underscored) name because it is a field
    # that must be written to the on-disk ``config.json`` alongside
    # every other hyperparameter. A leading underscore would make
    # the JSON key awkward (``_config_version``) and break any
    # downstream tooling that does ``json.load(...)["config_version"]``.
    config_version: str = "1"
    feature_dims: FeatureDims = field(default_factory=FeatureDims)
    hidden: HiddenDims = field(default_factory=HiddenDims)
    atom_blocks: AtomBlockConfig = field(default_factory=AtomBlockConfig)
    pairformer: PairformerConfig = field(default_factory=PairformerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    # Precision policy name. ``"reference"`` mirrors the reference bundle's
    # mixed-precision boundary (bf16 trunk/confidence with fp32 diffusion and
    # other preserved fp32 parameters); ``"float32"`` keeps the MLX port in
    # full precision throughout.
    compute_dtype: str = "reference"
    supported_token_sizes: Sequence[int] = (256, 384, 512, 768, 1024, 1536, 2048)
    template_restype_vocab: int = 33
    template_restype_embed_dim: int = 32
    num_rbf_radii: int = 6
    distance_rbf_scale: float = 4.8
    pocket_rbf_scale: float = 2.8
    attention_mask_value: float = -10000.0
    layer_norm_eps: float = 1e-5
    centroid_eps: float = 1e-4
    diffusion_sqrt_eps: float = 1e-6
    pairwise_distance_eps: float = 1e-10
    template_unit_vector_eps: float = 1e-12
    rigid_eps: float = 1e-8

    @property
    def token_single_dim(self) -> int:
        return self.hidden.token_single

    @property
    def token_pair_dim(self) -> int:
        return self.hidden.token_pair

    @property
    def diffusion_dim(self) -> int:
        return self.hidden.diffusion
