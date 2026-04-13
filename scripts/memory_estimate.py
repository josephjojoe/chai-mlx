"""Estimate peak memory usage for Chai-1 MLX inference at various sequence lengths.

Usage::

    python3 scripts/memory_estimate.py

Prints a table comparing bf16 vs fp32 peak memory at different token counts.
"""

from __future__ import annotations


def estimate_memory(
    N: int,
    dtype_bytes: int,
    num_samples: int = 5,
) -> dict[str, float]:
    """Return memory estimates in GB for a given token count N.

    The atom count is N * atom_multiplier (23 atoms/token for Chai-1).
    """
    A = N * 23  # atom count

    # ── Dimensions from ChaiConfig defaults ──
    single_dim = 384
    pair_dim = 256
    diffusion_dim = 768
    atom_single_dim = 128
    atom_pair_dim = 16
    msa_dim = 64
    template_pair_dim = 64

    num_pairformer_blocks = 48
    num_diffusion_blocks = 16
    num_diff_heads = 16
    diff_head_dim = 48

    tri_heads = 4
    tri_head_dim = 64
    single_heads = 16
    single_head_dim = 24

    query_block = 32
    kv_block = 128
    local_blocks = 3

    B = 1  # batch size

    d = dtype_bytes  # 2 for bf16, 4 for fp32

    # ── Weight memory ──
    # ~316M params, stored in dtype
    weight_params = 315_565_499
    weight_mem = weight_params * d

    # ── Activation memory (per-stage peaks) ──
    # We estimate the peak within each stage since MLX evaluates eagerly
    # and intermediates from prior stages can be freed.

    # --- Embedding stage ---
    # single: [B, N, 384], pair: [B, N, N, 256]
    # msa: [B, M, N, 64] (M ~ 16 typical MSA depth)
    # templates: [B, 4, N, N, 76] (input features)
    M = 16  # typical MSA depth
    T = 4   # max templates
    emb_single = B * N * single_dim * d
    emb_pair = B * N * N * pair_dim * d
    emb_msa = B * M * N * msa_dim * d
    emb_templates = B * T * N * N * template_pair_dim * d
    embedding_mem = emb_single + emb_pair + emb_msa + emb_templates

    # --- Template pairformer (2 blocks, pair_dim=64) ---
    # Processes each template independently: [B, N, N, 64]
    # Triangle mult intermediate: ~4x pair tensor (chunked)
    template_peak = B * N * N * template_pair_dim * d * 6

    # --- MSA module ---
    # Outer product mean: [B, M, N, 2, 8, 8] projection + [B, N, N, 512] intermediate
    # Pair-weighted averaging: weights [B, 8, N, N] + values
    msa_peak = (
        B * M * N * 128 * d +  # msa projections
        B * N * N * 512 * d +  # outer product intermediate
        B * N * N * pair_dim * d * 4  # pair + intermediates
    )

    # --- Pairformer stack (48 blocks) ---
    # This is the main memory consumer. Per block, the peak is during
    # triangle operations. We process one block at a time (mx.eval between blocks).
    #
    # Resident: single [B,N,384] + pair [B,N,N,256]
    # Per-block peak additions:
    #   Triangle mult (chunked, chunk_size=32):
    #     - z_normed: [B, N, N, 256]
    #     - a1, b1, a2, b2: [B, N, N, 32] each (4 tensors)
    #     - einsum output chunk: [B, N, N, 32] (2 tensors)
    #     - gate: [B, N, N, 256]
    #     Total extra: ~256+32*6+256 = ~704 channels * B*N*N*d
    #   Triangle attention (row-chunked, chunk=32):
    #     - z_normed: [B, N, N, pair_dim]
    #     - bias: [B, H, N, N] = [B, 4, N, N]
    #     - per-chunk: q,k,v,g: [B*32, H, N, D] = [B*32, 4, N, 64]
    #     - attn weights: [B*32, H, N, N]
    #   Transition: norm + up [B,N,N,2*2*256=1024] + down
    #   AttentionPairBias: q,k,v,g over single [B,N,16*24=384] + bias [B,16,N,N]

    pair_resident = B * N * N * pair_dim * d
    single_resident = B * N * single_dim * d

    # Triangle mult peak (on top of resident pair)
    tri_mult_extra = B * N * N * d * (
        pair_dim +      # z_normed
        32 * 4 +        # a1, b1, a2, b2 (one chunk)
        32 * 2 +        # einsum outputs (one chunk)
        pair_dim +      # x_out or x_in accumulated
        pair_dim         # gate
    )

    # Triangle attention peak (row-chunked)
    row_chunk = 32
    tri_attn_extra = (
        B * N * N * pair_dim * d +                              # z_normed
        B * tri_heads * N * N * d +                              # bias (both dirs)
        B * row_chunk * tri_heads * N * tri_head_dim * d * 4 +  # q,k,v,g per chunk
        B * row_chunk * tri_heads * N * N * d                    # attn scores per chunk
    )

    # Transition pair peak
    trans_pair_extra = B * N * N * (2 * 2 * pair_dim) * d  # up projection (expansion=2, SwiGLU doubles)

    # AttentionPairBias peak
    attn_extra = (
        B * N * (4 * single_heads * single_head_dim) * d +  # qkvg
        B * single_heads * N * N * d                          # attention scores
    )

    # Transition single
    trans_single_extra = B * N * (2 * 2 * single_dim) * d

    # Peak per pairformer block (they don't overlap — sequential ops within a block)
    pairformer_block_peak = pair_resident + single_resident + max(
        tri_mult_extra + trans_pair_extra,  # tri_mult and transition_pair run on same input
        tri_attn_extra,
        attn_extra + trans_single_extra,
    )

    # --- Diffusion module (per step) ---
    # 200 steps × num_samples (default 5), but per-step peak:
    # coords: [B, S, A, 3] fp32 always
    # s_cond: [B, S, N, 384]
    # x: [B, S, N, 768]
    # pair_biases: 16 × [B, 16, N, N] (precomputed, resident)
    # atom attention: local blocks with [B*S, num_blocks, query_block, kv_block, atom_pair_dim]
    S = num_samples
    coords_mem = B * S * A * 3 * 4  # always fp32

    diff_s_cond = B * S * N * single_dim * d
    diff_x = B * S * N * diffusion_dim * d
    diff_pair_biases = num_diffusion_blocks * B * num_diff_heads * N * N * d

    # Per diffusion transformer block:
    diff_attn_extra = (
        B * S * N * (3 * num_diff_heads * diff_head_dim) * d +  # qkv
        B * S * num_diff_heads * N * N * d                        # attn scores
    )
    diff_trans_extra = B * S * N * (2 * 2 * diffusion_dim) * d  # conditioned transition

    # Atom attention encoder/decoder (local attention blocks)
    # query_block=32, kv_block=128, atom_multiplier=23
    num_token_blocks = (N + query_block - 1) // query_block
    atom_attn_extra = (
        B * S * A * atom_single_dim * d +  # atom representations
        B * S * num_token_blocks * query_block * kv_block * atom_pair_dim * d +  # blocked pairs
        B * S * num_token_blocks * 4 * 32 * kv_block * d * local_blocks  # q,k,v per local block
    )

    diffusion_cache_mem = diff_pair_biases + B * N * single_dim * d + B * N * N * pair_dim * d
    diffusion_step_peak = (
        coords_mem +
        diff_s_cond +
        diff_x +
        diffusion_cache_mem +
        max(diff_attn_extra, diff_trans_extra, atom_attn_extra)
    )

    # Second-order diffusion step doubles the denoise cost (two forward passes)
    # but coords are small; the main overhead is the transformer pass
    diffusion_total_peak = diffusion_step_peak  # peak is per-step, not cumulative

    # --- Confidence head (4 pairformer blocks, pair_dim=256) ---
    # Similar to trunk pairformer but only 4 blocks
    # Plus: logit heads for pLDDT, PAE, PDE
    confidence_peak = pair_resident + single_resident + max(
        tri_mult_extra, tri_attn_extra, trans_pair_extra
    )

    # ── Overall peak ──
    # Stages are sequential, so peak is max across stages + weights
    trunk_peak = max(
        embedding_mem,
        template_peak,
        msa_peak,
        pairformer_block_peak,
    )
    overall_peak = weight_mem + max(trunk_peak, diffusion_total_peak, confidence_peak)

    # MLX overhead: graph allocations, eval buffers, ~10-20% extra
    mlx_overhead_factor = 1.15

    return {
        "N": N,
        "A": A,
        "dtype": "bf16" if d == 2 else "fp32",
        "weights_gb": weight_mem / 1e9,
        "trunk_peak_gb": trunk_peak / 1e9,
        "diffusion_peak_gb": diffusion_total_peak / 1e9,
        "confidence_peak_gb": confidence_peak / 1e9,
        "raw_peak_gb": overall_peak / 1e9,
        "estimated_peak_gb": overall_peak * mlx_overhead_factor / 1e9,
        "pair_tensor_gb": pair_resident / 1e9,
    }


def main() -> None:
    token_counts = [256, 384, 512, 768, 1024, 1536, 2048]
    num_samples = 5

    print("=" * 100)
    print(f"Chai-1 MLX Peak Memory Estimates (batch=1, samples={num_samples})")
    print("=" * 100)

    print(f"\n{'Tokens':>7} {'Atoms':>7} {'Mode':>5} {'Weights':>9} {'Pair[N²]':>9} "
          f"{'Trunk':>9} {'Diffusion':>9} {'Confid.':>9} {'Peak+OH':>10} {'Fits 16GB?':>11}")
    print("-" * 100)

    for N in token_counts:
        for d, label in [(2, "bf16"), (4, "fp32")]:
            r = estimate_memory(N, d, num_samples=num_samples)
            fits = "YES" if r["estimated_peak_gb"] < 14.5 else "NO"  # ~14.5 usable on 16GB
            print(f"{N:>7} {r['A']:>7} {label:>5} {r['weights_gb']:>8.2f}G "
                  f"{r['pair_tensor_gb']:>8.3f}G "
                  f"{r['trunk_peak_gb']:>8.2f}G {r['diffusion_peak_gb']:>8.2f}G "
                  f"{r['confidence_peak_gb']:>8.2f}G {r['estimated_peak_gb']:>9.2f}G "
                  f"{'  ✓' if fits == 'YES' else '  ✗':>11}")
        print()

    print("\nNotes:")
    print("  - 'Peak+OH' includes 15% overhead for MLX graph allocations and eval buffers")
    print("  - 'Fits 16GB?' assumes ~14.5 GB usable (OS + display use ~1.5 GB)")
    print("  - Diffusion runs 200 steps × 2 (second-order) = 400 denoise passes")
    print("  - The pair tensor [B,N,N,256] dominates — it scales as O(N²)")
    print("  - Trunk is typically the peak stage due to pairformer intermediates")
    print("  - BF16 halves both weights AND activations, giving ~2× memory savings")


if __name__ == "__main__":
    main()
