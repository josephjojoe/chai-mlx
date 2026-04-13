"""TorchScript-to-MLX parameter name mapping for Chai-1 weights.

Derived empirically by dumping ``named_parameters()`` from each TorchScript
``.pt`` file and aligning against ``ChaiMLX`` parameter paths.
"""

from __future__ import annotations

from typing import Sequence


# ===================================================================
# Reusable sub-component mapping builders
# ===================================================================

def _transition_map(src: str, dst: str) -> dict[str, str]:
    """SwiGLU transition: layer_norm + linear_no_bias_ab + linear_out (no bias)."""
    return {
        f"{src}.layer_norm.weight": f"{dst}.norm.weight",
        f"{src}.layer_norm.bias": f"{dst}.norm.bias",
        f"{src}.linear_no_bias_ab.weight": f"{dst}.up.weight",
        f"{src}.linear_out.weight": f"{dst}.down.weight",
    }


def _conditioned_transition_map(src: str, dst: str) -> dict[str, str]:
    """AdaLN conditioned transition."""
    return {
        f"{src}.ada_ln.lin_s_merged.weight": f"{dst}.adaln.to_scale_shift.weight",
        f"{src}.linear_a_nobias_double.weight": f"{dst}.up.weight",
        f"{src}.linear_b_nobias.weight": f"{dst}.down.weight",
        f"{src}.linear_s_biasinit_m2.weight": f"{dst}.gate.weight",
        f"{src}.linear_s_biasinit_m2.bias": f"{dst}.gate.bias",
    }


def _local_attn_block_map(src: str, dst: str) -> dict[str, str]:
    """LocalAttentionPairBiasBlock in atom attention."""
    return {
        f"{src}.single_layer_norm.lin_s_merged.weight": f"{dst}.adaln.to_scale_shift.weight",
        f"{src}.to_qkv.weight": f"{dst}.to_qkv.weight",
        f"{src}.q_bias": f"{dst}.q_bias",
        f"{src}.out_proj.weight": f"{dst}.output_proj.weight",
        f"{src}.out_proj.bias": f"{dst}.output_proj.bias",
    }


def _local_atom_transformer_map(src: str, dst: str, num_blocks: int = 3) -> dict[str, str]:
    m: dict[str, str] = {}
    ts = f"{src}.local_diffn_transformer"
    # blocked_pairs2blocked_bias: Sequential(LayerNorm[0], Linear[1])
    m[f"{ts}.blocked_pairs2blocked_bias.0.weight"] = f"{dst}.pair_norm.weight"
    m[f"{ts}.blocked_pairs2blocked_bias.0.bias"] = f"{dst}.pair_norm.bias"
    m[f"{ts}.blocked_pairs2blocked_bias.1.weight"] = f"{dst}.blocked_pairs2blocked_bias.weight"
    for i in range(num_blocks):
        m.update(_local_attn_block_map(f"{ts}.local_attentions.{i}", f"{dst}.attn_blocks.{i}"))
        m.update(_conditioned_transition_map(f"{ts}.transitions.{i}", f"{dst}.transitions.{i}"))
    return m


def _pair_update_block_map(src: str, dst: str) -> dict[str, str]:
    """PairUpdateBlock: proj_h/proj_w are Sequential(ReLU[0], Linear[1])."""
    return {
        f"{src}.atom_single_to_atom_pair_proj_h.1.weight": f"{dst}.proj_h.weight",
        f"{src}.atom_single_to_atom_pair_proj_w.1.weight": f"{dst}.proj_w.weight",
        f"{src}.atom_pair_mlp.0.weight": f"{dst}.mlp.fc1.weight",
        f"{src}.atom_pair_mlp.2.weight": f"{dst}.mlp.fc2.weight",
    }


def _triangle_mult_map(src: str, dst: str) -> dict[str, str]:
    """TriangleMultiplication. layernorm_out/layernorm_in are affine=False (no params)."""
    return {
        f"{src}.layernorm_z_in.weight": f"{dst}.layernorm_z_in.weight",
        f"{src}.layernorm_z_in.bias": f"{dst}.layernorm_z_in.bias",
        f"{src}.merged_linear_p.weight": f"{dst}.merged_linear_p.weight",
        f"{src}.merged_linear_g.weight": f"{dst}.merged_linear_g.weight",
        f"{src}.linear_z_out.weight": f"{dst}.linear_z_out.weight",
    }


def _triangle_attn_map(src: str, dst: str) -> dict[str, str]:
    """TriangleAttention v2a. pair_norm is affine=False (no learnable params)."""
    return {
        f"{src}.pair2b.weight": f"{dst}.pair2b.weight",
        f"{src}.pair2qkvg1.weight": f"{dst}.pair2qkvg1.weight",
        f"{src}.pair2qkvg2.weight": f"{dst}.pair2qkvg2.weight",
        f"{src}.linear_out.weight": f"{dst}.linear_out.weight",
        f"{src}.out_scalers": f"{dst}.out_scalers",
    }


def _confidence_tri_attn_map(src: str, dst: str) -> dict[str, str]:
    """ConfidenceTriangleAttention (fused single-projection)."""
    return {
        f"{src}.pair_layer_norm.weight": f"{dst}.pair_norm.weight",
        f"{src}.pair_layer_norm.bias": f"{dst}.pair_norm.bias",
        f"{src}.pair2qkvgb.weight": f"{dst}.pair2qkvgb.weight",
        f"{src}.linear_out.weight": f"{dst}.linear_out.weight",
    }


def _attention_pair_bias_map(src: str, dst: str) -> dict[str, str]:
    """AttentionPairBias (trunk/confidence pairformer)."""
    return {
        f"{src}.single_layer_norm.weight": f"{dst}.single_norm.weight",
        f"{src}.single_layer_norm.bias": f"{dst}.single_norm.bias",
        f"{src}.pair_layer_norm.weight": f"{dst}.pair_norm.weight",
        f"{src}.pair_layer_norm.bias": f"{dst}.pair_norm.bias",
        f"{src}.pair_linear.weight": f"{dst}.pair_linear.weight",
        f"{src}.attention.input2qkvg.weight": f"{dst}.input2qkvg.weight",
        f"{src}.attention.query_bias": f"{dst}.query_bias",
        f"{src}.attention.output_proj.weight": f"{dst}.output_proj.weight",
    }


def _pairformer_block_map(
    src: str, dst: str, *, has_single: bool = True, fused_triangle: bool = False,
) -> dict[str, str]:
    m: dict[str, str] = {}
    m.update(_triangle_mult_map(f"{src}.triangle_multiplication", f"{dst}.triangle_multiplication"))
    if fused_triangle:
        m.update(_confidence_tri_attn_map(f"{src}.triangle_attention", f"{dst}.triangle_attention"))
    else:
        m.update(_triangle_attn_map(f"{src}.triangle_attention", f"{dst}.triangle_attention"))
    # TorchScript uses transition_pair (not pair_transition)
    m.update(_transition_map(f"{src}.transition_pair", f"{dst}.transition_pair"))
    if has_single:
        m.update(_attention_pair_bias_map(f"{src}.attention_pair_bias", f"{dst}.attention_pair_bias"))
        m.update(_transition_map(f"{src}.transition_single", f"{dst}.transition_single"))
    return m


# ===================================================================
# Component-level maps
# ===================================================================

def _feature_embedding_map() -> dict[str, str]:
    prefix = "input_embedder.feature_embedding."
    m: dict[str, str] = {}
    for ts_key, mlx_key in [
        ("input_projs.TOKEN.0", "token_proj"),
        ("input_projs.TOKEN_PAIR.0", "token_pair_proj"),
        ("input_projs.ATOM.0", "atom_proj"),
        ("input_projs.ATOM_PAIR.0", "atom_pair_proj"),
        ("input_projs.MSA.0", "msa_proj"),
        ("input_projs.TEMPLATES.0", "template_proj"),
    ]:
        m[f"{ts_key}.weight"] = f"{prefix}{mlx_key}.weight"
        m[f"{ts_key}.bias"] = f"{prefix}{mlx_key}.bias"
    m["feature_embeddings.TEMPLATES.TemplateResType.embedding.weight"] = (
        f"{prefix}template_restype_embedding.weight"
    )
    m["feature_embeddings.TOKEN_PAIR.TokenDistanceRestraint.radii"] = (
        f"{prefix}distance_restraint_radii"
    )
    m["feature_embeddings.TOKEN_PAIR.TokenPairPocketRestraint.radii"] = (
        f"{prefix}pocket_restraint_radii"
    )
    return m


def _bond_loss_input_proj_map() -> dict[str, str]:
    return {"weight": "input_embedder.bond_projection.proj.weight"}


def _token_embedder_map() -> dict[str, str]:
    ts = "token_single_input_emb"
    dst = "input_embedder.token_input"
    m: dict[str, str] = {}

    # Atom encoder
    enc_s = f"{ts}.atom_encoder"
    enc_d = f"{dst}.atom_encoder"
    m[f"{enc_s}.to_atom_cond.weight"] = f"{enc_d}.to_atom_cond.weight"
    m.update(_pair_update_block_map(f"{enc_s}.pair_update_block", f"{enc_d}.pair_update_block"))
    m.update(_local_atom_transformer_map(f"{enc_s}.atom_transformer", f"{enc_d}.atom_transformer"))
    # to_token_single: Sequential(Linear[0](no bias), ReLU[1])
    # .0 = Linear(128, 384, no bias), .1 = ReLU (no params)
    # Verified from token_embedder_code.txt: .0 is torch.linear(input, weight), .1 is relu_.
    m[f"{enc_s}.to_token_single.0.weight"] = f"{enc_d}.to_token_single.weight"

    # Top-level projections
    m["token_single_proj_in_trunk.weight"] = f"{dst}.token_single_proj_in_trunk.weight"
    m["token_single_proj_in_structure.weight"] = f"{dst}.token_single_proj_in_structure.weight"
    m["token_single_to_token_pair_outer_sum_proj.weight"] = f"{dst}.single_to_pair_proj.weight"
    m["token_pair_proj_in_trunk.weight"] = f"{dst}.token_pair_proj_in_trunk.weight"
    return m


def _trunk_map() -> dict[str, str]:
    dst = "trunk_module"
    m: dict[str, str] = {}

    # Recycle projections: Sequential(LayerNorm[0], Linear[1])
    for path in ("token_single_recycle_proj", "token_pair_recycle_proj"):
        m[f"{path}.0.weight"] = f"{dst}.{path}.norm.weight"
        m[f"{path}.0.bias"] = f"{dst}.{path}.norm.bias"
        m[f"{path}.1.weight"] = f"{dst}.{path}.proj.weight"

    # Template embedder
    te_s = "template_embedder"
    te_d = f"{dst}.template_embedder"
    # proj_in: Sequential(LayerNorm[0], Linear[1])
    m[f"{te_s}.proj_in.0.weight"] = f"{te_d}.proj_in_norm.weight"
    m[f"{te_s}.proj_in.0.bias"] = f"{te_d}.proj_in_norm.bias"
    m[f"{te_s}.proj_in.1.weight"] = f"{te_d}.proj_in.weight"
    m[f"{te_s}.template_layernorm.weight"] = f"{te_d}.template_layernorm.weight"
    m[f"{te_s}.template_layernorm.bias"] = f"{te_d}.template_layernorm.bias"
    # proj_out: Sequential(ReLU[0], Linear[1])
    m[f"{te_s}.proj_out.1.weight"] = f"{te_d}.proj_out.weight"
    # Template pairformer (2 blocks, pair-only)
    for i in range(2):
        m.update(_pairformer_block_map(
            f"{te_s}.pairformer.blocks.{i}", f"{te_d}.blocks.{i}", has_single=False,
        ))

    # MSA module
    ms_s = "msa_module"
    ms_d = f"{dst}.msa_module"
    m[f"{ms_s}.linear_s2m.weight"] = f"{ms_d}.linear_s2m.weight"

    for i in range(4):  # outer_product_mean
        s, d = f"{ms_s}.outer_product_mean.{i}", f"{ms_d}.outer_product_mean.{i}"
        m[f"{s}.weight_ab"] = f"{d}.weight_ab"
        m[f"{s}.ln_out.weight"] = f"{d}.ln_out.weight"
        m[f"{s}.ln_out.bias"] = f"{d}.ln_out.bias"
        m[f"{s}.linear_out.weight"] = f"{d}.linear_out.weight"
        m[f"{s}.linear_out.bias"] = f"{d}.linear_out.bias"

    for i in range(3):  # msa_pair_weighted_averaging
        s = f"{ms_s}.msa_pair_weighted_averaging.{i}"
        d = f"{ms_d}.msa_pair_weighted_averaging.{i}"
        for suffix in ("layernorm_msa.weight", "layernorm_msa.bias",
                        "linear_msa2vg.weight", "layernorm_pair.weight",
                        "layernorm_pair.bias", "linear_pair.weight",
                        "linear_out_no_bias.weight"):
            m[f"{s}.{suffix}"] = f"{d}.{suffix}"

    for i in range(3):  # msa_transition
        m.update(_transition_map(f"{ms_s}.msa_transition.{i}", f"{ms_d}.msa_transition.{i}"))

    for i in range(4):  # pair_transition
        m.update(_transition_map(f"{ms_s}.pair_transition.{i}", f"{ms_d}.pair_transition.{i}"))

    for i in range(4):  # triangular_multiplication
        m.update(_triangle_mult_map(
            f"{ms_s}.triangular_multiplication.{i}", f"{ms_d}.triangular_multiplication.{i}",
        ))

    for i in range(4):  # triangular_attention
        m.update(_triangle_attn_map(
            f"{ms_s}.triangular_attention.{i}", f"{ms_d}.triangular_attention.{i}",
        ))

    # Pairformer stack (48 blocks)
    for i in range(48):
        m.update(_pairformer_block_map(
            f"pairformer_stack.blocks.{i}", f"{dst}.pairformer_stack.blocks.{i}", has_single=True,
        ))

    return m


def _diffusion_module_map() -> dict[str, str]:
    dst = "diffusion_module"
    m: dict[str, str] = {}

    # Diffusion conditioning
    dc_s = "diffusion_conditioning"
    dc_d = f"{dst}.diffusion_conditioning"
    # token_pair_proj: Sequential(LayerNorm[0], Linear[1])
    m[f"{dc_s}.token_pair_proj.0.weight"] = f"{dc_d}.token_pair_norm.weight"
    m[f"{dc_s}.token_pair_proj.0.bias"] = f"{dc_d}.token_pair_norm.bias"
    m[f"{dc_s}.token_pair_proj.1.weight"] = f"{dc_d}.token_pair_proj.weight"
    # token_in_proj: Sequential(LayerNorm[0], Linear[1])
    m[f"{dc_s}.token_in_proj.0.weight"] = f"{dc_d}.token_in_norm.weight"
    m[f"{dc_s}.token_in_proj.0.bias"] = f"{dc_d}.token_in_norm.bias"
    m[f"{dc_s}.token_in_proj.1.weight"] = f"{dc_d}.token_in_proj.weight"

    m.update(_transition_map(f"{dc_s}.single_trans1", f"{dc_d}.single_trans1"))
    m.update(_transition_map(f"{dc_s}.pair_trans1", f"{dc_d}.pair_trans1"))
    m.update(_transition_map(f"{dc_s}.single_trans2", f"{dc_d}.single_trans2"))
    m.update(_transition_map(f"{dc_s}.pair_trans2", f"{dc_d}.pair_trans2"))

    # Fourier embedding (note: TorchScript uses "weights" plural)
    m[f"{dc_s}.fourier_embedding.weights"] = f"{dc_d}.fourier_embedding.weights"
    m[f"{dc_s}.fourier_embedding.bias"] = f"{dc_d}.fourier_embedding.bias"
    # fourier_proj: Sequential(LayerNorm[0], Linear[1])
    m[f"{dc_s}.fourier_proj.0.weight"] = f"{dc_d}.fourier_proj_norm.weight"
    m[f"{dc_s}.fourier_proj.0.bias"] = f"{dc_d}.fourier_proj_norm.bias"
    m[f"{dc_s}.fourier_proj.1.weight"] = f"{dc_d}.fourier_proj.weight"

    m[f"{dc_s}.single_ln.weight"] = f"{dc_d}.single_ln.weight"
    m[f"{dc_s}.single_ln.bias"] = f"{dc_d}.single_ln.bias"
    m[f"{dc_s}.pair_ln.weight"] = f"{dc_d}.pair_ln.weight"
    m[f"{dc_s}.pair_ln.bias"] = f"{dc_d}.pair_ln.bias"

    # Atom attention encoder
    enc_s = "atom_attention_encoder"
    enc_d = f"{dst}.atom_attention_encoder"
    m[f"{enc_s}.to_atom_cond.weight"] = f"{enc_d}.to_atom_cond.weight"
    # token_to_atom_single: Sequential(LayerNorm[0], Linear[1])
    m[f"{enc_s}.token_to_atom_single.0.weight"] = f"{enc_d}.token_to_atom_single_norm.weight"
    m[f"{enc_s}.token_to_atom_single.0.bias"] = f"{enc_d}.token_to_atom_single_norm.bias"
    m[f"{enc_s}.token_to_atom_single.1.weight"] = f"{enc_d}.token_to_atom_single.weight"
    m[f"{enc_s}.prev_pos_embed.weight"] = f"{enc_d}.prev_pos_embed.weight"
    m.update(_pair_update_block_map(f"{enc_s}.pair_update_block", f"{enc_d}.pair_update_block"))
    m.update(_local_atom_transformer_map(f"{enc_s}.atom_transformer", f"{enc_d}.atom_transformer"))
    # to_token_single: Sequential(Linear[0](no bias), ReLU[1])
    m[f"{enc_s}.to_token_single.0.weight"] = f"{enc_d}.to_token_single.weight"
    # token_pair_to_atom_pair: Sequential(LayerNorm[0], Linear[1])
    m[f"{enc_s}.token_pair_to_atom_pair.0.weight"] = f"{dst}.token_pair_to_atom_pair_norm.weight"
    m[f"{enc_s}.token_pair_to_atom_pair.0.bias"] = f"{dst}.token_pair_to_atom_pair_norm.bias"
    m[f"{enc_s}.token_pair_to_atom_pair.1.weight"] = f"{dst}.token_pair_to_atom_pair.weight"

    # Diffusion transformer (16 blocks) — flat structure, not nested under "attention"
    for i in range(16):
        bs = f"diffusion_transformer.blocks.{i}"
        bd = f"{dst}.diffusion_transformer.blocks.{i}"
        # Attention (flat in TorchScript)
        m[f"{bs}.norm_in.lin_s_merged.weight"] = f"{bd}.attn.adaln.to_scale_shift.weight"
        m[f"{bs}.pair_layer_norm.weight"] = f"{bd}.attn.pair_norm.weight"
        m[f"{bs}.pair_layer_norm.bias"] = f"{bd}.attn.pair_norm.bias"
        m[f"{bs}.pair_linear.weight"] = f"{bd}.attn.pair_linear.weight"
        m[f"{bs}.to_qkv.weight"] = f"{bd}.attn.to_qkv.weight"
        m[f"{bs}.q_bias"] = f"{bd}.attn.query_bias"
        m[f"{bs}.to_out.weight"] = f"{bd}.attn.to_out.weight"
        # gate_proj: Sequential(Linear[0]) in TorchScript
        m[f"{bs}.gate_proj.0.weight"] = f"{bd}.attn.gate_proj.weight"
        m[f"{bs}.gate_proj.0.bias"] = f"{bd}.attn.gate_proj.bias"
        # Transition
        m.update(_conditioned_transition_map(f"{bs}.transition", f"{bd}.transition"))

    # Atom attention decoder
    dec_s = "atom_attention_decoder"
    dec_d = f"{dst}.atom_attention_decoder"
    m[f"{dec_s}.token_to_atom.weight"] = f"{dec_d}.token_to_atom.weight"
    m.update(_local_atom_transformer_map(f"{dec_s}.atom_transformer", f"{dec_d}.atom_transformer"))
    # to_pos_updates: Sequential(LayerNorm[0], Linear[1])
    m[f"{dec_s}.to_pos_updates.0.weight"] = f"{dec_d}.output_norm.weight"
    m[f"{dec_s}.to_pos_updates.0.bias"] = f"{dec_d}.output_norm.bias"
    m[f"{dec_s}.to_pos_updates.1.weight"] = f"{dec_d}.to_pos_updates.weight"

    # Misc projections
    m["structure_cond_to_token_structure_proj.weight"] = f"{dst}.structure_cond_to_token_structure_proj.weight"
    m["post_attn_layernorm.weight"] = f"{dst}.post_attn_layernorm.weight"
    m["post_attn_layernorm.bias"] = f"{dst}.post_attn_layernorm.bias"
    m["post_atom_cond_layernorm.weight"] = f"{dst}.post_atom_cond_layernorm.weight"
    m["post_atom_cond_layernorm.bias"] = f"{dst}.post_atom_cond_layernorm.bias"

    return m


def _confidence_head_map() -> dict[str, str]:
    dst = "confidence_head"
    m: dict[str, str] = {}

    m["single_to_pair_proj.weight"] = f"{dst}.single_to_pair_proj.weight"
    m["atom_distance_bins_projection.weight"] = f"{dst}.atom_distance_bins_projection.weight"

    for i in range(4):
        m.update(_pairformer_block_map(
            f"blocks.{i}", f"{dst}.blocks.{i}", has_single=True, fused_triangle=True,
        ))

    m["plddt_projection.weight"] = f"{dst}.plddt_projection.weight"
    m["pae_projection.weight"] = f"{dst}.pae_projection.weight"
    m["pde_projection.weight"] = f"{dst}.pde_projection.weight"

    return m


# ===================================================================
# Public API
# ===================================================================

COMPONENT_BUILDERS: dict[str, callable] = {
    "feature_embedding": _feature_embedding_map,
    "bond_loss_input_proj": _bond_loss_input_proj_map,
    "token_embedder": _token_embedder_map,
    "trunk": _trunk_map,
    "diffusion_module": _diffusion_module_map,
    "confidence_head": _confidence_head_map,
}


def build_rename_map(component: str) -> dict[str, str]:
    builder = COMPONENT_BUILDERS.get(component)
    if builder is None:
        raise ValueError(f"Unknown component {component!r}. Valid: {sorted(COMPONENT_BUILDERS)}")
    return builder()


def build_full_rename_map() -> dict[str, str]:
    full: dict[str, str] = {}
    for name in COMPONENT_BUILDERS:
        full.update(build_rename_map(name))
    return full


def reshape_einsum_weight(mlx_key: str, arr: "np.ndarray") -> "np.ndarray":
    """Reshape multi-dimensional einsum weights to 2D nn.Linear format.

    TorchScript stores several attention weights as multi-dimensional tensors
    consumed via ``torch.einsum``, while the MLX port uses standard
    ``nn.Linear`` (2D weight matrices).  This function converts the former
    to the latter, preserving numerical equivalence.

    Returns the array unchanged when no reshape is needed (ndim <= 2 or
    not a recognized einsum weight).
    """
    # RBF radii: TorchScript stores [1, num_radii], MLX expects (num_radii,)
    if mlx_key.endswith("_radii") and arr.ndim == 2 and arr.shape[0] == 1:
        return arr.squeeze(0)

    if arr.ndim <= 2:
        return arr

    # OPM weight_ab [2, 8, 8, msa_dim] is intentionally 4D (consumed via mx.einsum).
    if "weight_ab" in mlx_key:
        return arr

    # AttentionPairBias.input2qkvg: [in_dim, 4, H, D] → [4*H*D, in_dim]
    # Trunk einsum:      "dfa,aebc->edbfc"  (a = contracted / in_dim, first)
    # Confidence einsum: "fae,ebcd->bfcad"  (e = contracted / in_dim, first)
    if "input2qkvg.weight" in mlx_key:
        in_dim = arr.shape[0]
        return arr.reshape(in_dim, -1).T

    # AttentionPairBias.output_proj: [H, D, out_dim] → [out_dim, H*D]
    # Trunk einsum:      "ecbd,cda->eba"  (c,d contracted; a = out_dim, last)
    # Confidence einsum: "cdae,deb->cab"  (d,e contracted; b = out_dim, last)
    if "attention_pair_bias.output_proj.weight" in mlx_key:
        out_dim = arr.shape[-1]
        return arr.reshape(-1, out_dim).T

    # LocalAttentionPairBiasBlock.to_qkv: [3, H, D, in_dim] → [3*H*D, in_dim]
    # Einsum: "eda,fbca->febdc"  (a = contracted / in_dim, last)
    if ".attn_blocks." in mlx_key and ".to_qkv.weight" in mlx_key:
        in_dim = arr.shape[-1]
        return arr.reshape(-1, in_dim)

    # LocalAtomTransformer.blocked_pairs2blocked_bias (EinMix):
    # [num_blocks, num_heads, pair_dim] → [num_blocks*num_heads, pair_dim]
    if "blocked_pairs2blocked_bias.weight" in mlx_key and arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])

    import warnings
    warnings.warn(
        f"Unexpected >2D weight at {mlx_key} with shape {arr.shape}; "
        f"leaving as-is (may cause shape mismatch at load time)"
    )
    return arr


def rename_state_dict(
    state: dict[str, object], rename_map: dict[str, str],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for old_key, value in state.items():
        new_key = rename_map.get(old_key)
        if new_key is not None:
            out[new_key] = value
        else:
            out[f"__unmapped__.{old_key}"] = value
    return out


def discover_mismatches(
    npz_keys: Sequence[str], mlx_keys: Sequence[str], rename_map: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    mapped_mlx = set()
    unmapped_npz: list[str] = []
    for k in npz_keys:
        if k in rename_map:
            mapped_mlx.add(rename_map[k])
        else:
            unmapped_npz.append(k)
    mlx_set = set(mlx_keys)
    missing_mlx = sorted(mlx_set - mapped_mlx)
    extra_mlx = sorted(mapped_mlx - mlx_set)
    return unmapped_npz, missing_mlx, extra_mlx
