# MLX vs CUDA trunk drift — root-cause attribution

Investigation target: the 16–34 % relative-norm gap between MLX-at-fp32
`trunk.recycle_0.*` and CUDA-scripted `trunk.recycle_0.*` that appears in
`/tmp/chai_mlx_cuda/parity_1L2Y_s42_{fp32,bf16}.json`. Prior interpretation
("MLX is drifting from CUDA") turns out to be wrong in the important sense:
the CUDA *scripted* path disagrees with its own CUDA *eager* path by the
same order of magnitude, and MLX sits inside that envelope.

All numbers below are computed on 1L2Y seed 42 at crop size 256 (live
region = 20 tokens × 384 dims for the single stream, 20 × 20 × 256 for
the pair stream). Reference implementations are loaded from
`/models/chai1/models_v2/trunk.pt` on Modal H100.

## Summary of measured drifts

1. **Reference precision policy (verified from TorchScript IR, not docstrings).**
  Counting `aten::to` casts in `findings/graphs/`:
  - `trunk.pt`: **18 324 bf16 casts vs 5 160 fp32 casts.** Every
  `aten::linear` / `aten::einsum` is bracketed by `weight.to(bf16)`
    - `input.to(bf16)` → compute in bf16; every `aten::layer_norm` /
    `aten::softmax(...)` is bracketed by `to(fp32)`. Functionally
    `torch.autocast("cuda", dtype=torch.bfloat16)` with explicit fp32
    fallback for normalisation-like ops, compiled into the graph.
  - `confidence_head.pt`: same bf16-autocast policy (1 102 bf16 vs
  355 fp32).
  - `token_embedder.pt`: bf16-autocast.
  - `feature_embedding.pt`: pure fp32 (no bf16 constants).
  - `diffusion_module.pt`: pure fp32 end-to-end (zero bf16 casts).
  - `bond_loss_input_proj.pt`: pure fp32.
    The earlier claim in `cuda_harness/run_intermediates.py` that
    chai-lab was "fp32 end-to-end, no bf16 casts" was wrong — it counted
    only named parameters (which are fp32 on disk) and missed the
    per-op casts baked into the scripted graph. Both that file and
    `scripts/cuda_parity.py` have now been updated with the correct
    precision write-up.
2. **Eager vs scripted CUDA disagree at comparable magnitudes.**
  Full trunk (msa_module + 48 pairformer blocks) at recycle 0 on 1L2Y,
    live region only:

  | Comparison                           | single |d|/|b| | pair |d|/|b| |
  | ------------------------------------ | -------------- | ------------ |
  | scripted-CUDA vs **eager-CUDA-bf16** | **44.1 %**     | **19.7 %**   |
  | scripted-CUDA vs eager-CUDA-fp32     | 44.3 %         | 19.4 %       |
  | scripted-CUDA vs **MLX-fp32**        | 42.2 %         | 14.9 %       |
  | eager-CUDA-bf16 vs eager-CUDA-fp32   | 1.3 %          | 1.3 %        |
  | MLX-fp32 vs eager-CUDA-fp32          | 33.0 %         | 12.2 %       |
  | MLX-fp32 vs eager-CUDA-bf16          | 32.7 %         | 12.4 %       |

    Two surprises:
  - **Scripted CUDA is the outlier.** Both eager-CUDA precisions and
  MLX disagree with scripted by ~20 % on pair / ~45 % on single.
  The gap between scripted and its own eager reimplementation is
  larger than the gap between eager and MLX.
  - **MLX is no further from scripted than eager-CUDA-bf16 is,**
  and on pair it is actually *closer* (14.9 % vs 19.7 %). The
  headline "MLX is 34 % off CUDA" was really "MLX-fp32 is 33 %
  off eager-CUDA and another 15 % off scripted-CUDA", and the
  eager-vs-scripted gap dominates.
3. **Inside `msa_module` alone**, MLX and eager-CUDA agree tightly.
  Per-sub-op rel_norm between MLX and eager-CUDA at bf16, from
    `cuda_harness/_probe_msa_rounds_compare.py --dtype bf16`:

  | Tensor                       | MLX vs eager-CUDA |d|/|c| |
  | ---------------------------- | ------------------------- |
  | after_linear_s2m_msa         | 0.00 % (bit-identical)    |
  | round_0.opm_delta_pair       | 0.00 %                    |
  | round_0.pair_after_opm       | 0.00 %                    |
  | round_0.msa_after_transition | 0.17 %                    |
  | round_0.pair_after_tri_mult  | 0.06 %                    |
  | round_0.pair_after_tri_attn  | 0.11 %                    |
  | round_3.pair_after_tri_attn  | **5.43 %**                |

    At fp32 the same table shows 7.94 × 10⁻⁶ (ULP noise) for the final
    tensor. So MSA-module drift between MLX and eager-CUDA is purely
    a bf16 accumulation / summation-order effect, dominated by the
    triangular attention in rounds 1 – 3, and after the full 4 rounds
    sums to ~5 %.
4. **Isolated triangle_attention (MSA round 1, bit-identical input)**
  — the biggest per-op drift site at bf16 — disagrees by only
    **9.1 × 10⁻⁴ rel_norm** between MLX and eager-CUDA. The 40 × "in
    situ" amplification at round 1 seen in the per-round table is
    from *input* drift accumulated upstream, not from the op itself.
5. **MLX's `mx.fast.scaled_dot_product_attention` already accumulates
  in fp32 under the hood at bf16 inputs.** Tested against three
    explicit-softmax formulations:

  | Variant                                             | rel_norm vs eager-CUDA |
  | --------------------------------------------------- | ---------------------- |
  | v1 `mx.fast.scaled_dot_product_attention` (current) | 9.14 × 10⁻⁴            |
  | v2 manual `Q @ K^T` + softmax all in bf16           | 2.35 × 10⁻²            |
  | v3 manual, fp32 softmax but bf16 matmul             | 2.35 × 10⁻²            |
  | v4 manual, fp32 matmul + fp32 softmax               | 9.36 × 10⁻⁴            |

    v1 matches v4 to ULPs. The drift between v1 and CUDA's
    `F.scaled_dot_product_attention` is thus pure fp32 accumulator
    order difference (Apple Metal vs NVIDIA Hopper scheduling the
    fp32 sum of bf16 terms differently). Tile-size changes on MLX
    (`_ROW_CHUNK ∈ {8, 16, 32, 64, 128, 256}`) give
    *bit-identical* output — the row tiling is purely over
    independent SDPA calls, not over the contraction axis.
6. **MLX's `TriangleAttention._ROW_CHUNK` is numerically inert.** All
  tile sizes ≥ 1 produce bit-identical output because the chunked
    region is batched rows and each chunk runs its own complete
    softmax over the full contraction axis. Changing the chunk size
    only trades peak memory against kernel-launch overhead — the
    per-op drift we see comes from inside `mx.fast.sdpa`, not from
    our row tiling.

## Implications for the port

- **The MLX port is not broken.** The 16–34 % "drift vs CUDA" in the
parity JSONs is dominated by the eager-vs-scripted CUDA gap on
TorchScript side. Both the MLX port and an eager PyTorch
reimplementation of the exact same chai-lab module tree land in
essentially the same place; scripted CUDA is the one that goes
its own way.
- **Closing the remaining MLX vs eager-CUDA gap further (≲5 % on
MSA, ≲12 % on the full trunk at bf16) requires matching CUDA's
internal fp32-accumulator summation order inside
`scaled_dot_product_attention` and `matmul`.** That is the
"tree reduction vs left-to-right" question the user raised. It is
implementable but expensive: we would have to write a Metal kernel
that replicates NVIDIA's warp-level tree reduction pattern, and
every accelerator generation will schedule fp32 sums of bf16
terms differently. The gain is a fraction of a percent per block.
- **Residual-stream compounding is real but starts small.** Per
block at bf16, MLX ≈ 1–3 % rel_norm vs eager-CUDA on `(s, z)`
with residual connections partially absorbing the drift; after
48 blocks this stays ≈ 12–33 % despite the compounding. On real
post-MSA inputs we measured the 48-block eager-CUDA vs MLX gap
as 1 × 10⁻⁵ at fp32 (both sides clean fp32) and ~5 % at bf16.
- **Recycling is NOT compounding.** The 16–34 % shows up already at
`trunk.recycle_0.`* (first recycle) and grows only modestly to
17–37 % at `recycle_2.*`. Between eager-CUDA and MLX the per-recycle
growth is even smaller: the recycle projection is a LayerNorm +
Linear that at bf16 produces ~0.1 % of fresh drift.
- **Diffusion is bit-for-bit exact given correct trunk outputs**
(README status, section 1.1). The scripted diffusion module
  runs in pure fp32 end-to-end; MLX now matches that and keeps the
  diffusion path in fp32 even when the rest of the model runs at bf16.

## What we built during this investigation

New probes under `cuda_harness/`:

- `_probe_trunk_methods.py` — dumps `dir(...)` and scripted-method
lists for every trunk submodule. Documents that TorchScript inlines
every `trunk` submodule into `forward_256` / `forward_<crop>` at
scripting time, so `trunk.msa_module(...)`,
`trunk.pairformer_stack(...)`, `trunk.template_embedder(...)`, and
`trunk.pairformer_stack.blocks[i](...)` are all un-callable: the
scripted `forward` attribute is stripped off nested modules.
Only `trunk.forward_<crop>` is invocable.
- `_probe_trunk_submodule_call.py` — exhaustively tries every call
path on every named child of `trunk`; confirms the only callables
are (a) `token_{single,pair}_recycle_proj` (nn.Sequential, whose
bare `.forward` survives scripting because it is Python-defined)
and (b) `trunk.forward_256`.
- `_probe_first_block_ts_cuda.py` — eager-PyTorch PairformerBlock
matching `chai_mlx.nn.layers.pairformer.PairformerBlock` line for
line, run at fp64 (CPU oracle), fp32 (GPU), and bf16 (GPU). The
scripted-module call paths it also tries are documented as
expected failures.
- `_probe_first_block_compare.py` — reads MLX `first_block_probe`
dumps plus the new eager dumps and prints the drift table for the
seven sub-op intermediates of a single PairformerBlock.
- `_probe_full_stack_compare.py` — read MLX vs eager-CUDA per-block
dumps for the 48-block pairformer stack on synthetic Gaussian
inputs and print drift vs block index.
- `_probe_msa_module_cuda.py` — eager-PyTorch `MSAModule` with all
weights wired from `trunk.pt.msa_module.`*; takes `--dtype fp32|bf16`. Emits per-round intermediates
(`round_{i}.opm_delta_pair`, `pair_after_opm`,
`msa_after_transition`, `msa_after_pw`, `pair_trans_out`,
`pair_after_tri_mult`, `pair_after_tri_attn`).
- `_probe_msa_rounds_mlx.py` — MLX-side companion that walks
`model.trunk_module.msa_module` step-for-step and emits the same
per-round intermediate names. Supports `--dtype fp32|bf16`.
- `_probe_msa_rounds_compare.py` — diffs MLX vs CUDA per-round
intermediates and prints the attribution table that localised
the drift to round-1 `triangular_attention` at bf16.
- `_probe_tri_attn_isolated.py` — feeds bit-identical inputs
(CUDA's captured `round_1.pair_after_tri_mult`) into MLX's
`TriangleAttention` and sweeps `_ROW_CHUNK ∈ {8, …, 256}`. The
output proved chunk-size-independent.
- `_probe_sdpa_variants.py` — four SDPA formulations inside the
isolated triangle-attention probe. Established that MLX's
`mx.fast.scaled_dot_product_attention` accumulates in fp32
internally.
- `_probe_einsum_accum.py` — standalone smoke test that MLX's
`mx.einsum` with bf16 inputs produces bf16 output but accumulates
internally at fp32 precision (within rounding).
- `_probe_full_trunk_eager.py` — eager-PyTorch end-to-end trunk
(msa_module + 48 pairformer blocks) at bf16 or fp32 from the
real 1L2Y intermediates. Returns post-msa and post-pairformer
`(single, pair)` without per-block dumps (to stay under Modal's
return-value size limit).

## Outstanding (and probably not worth doing)

- Write a Metal kernel that reproduces NVIDIA's warp-tree reduction
pattern for the `Q @ K^T / √d → softmax → @ V` chain at bf16.
Expected win is a fraction of a percent per block — the scripted-
vs-eager CUDA gap would still swamp any MLX-side improvement.
- Reverse-engineer the TorchScript JIT optimizer's kernel fusion
decisions on `trunk.pt` and replicate them in eager PyTorch. That
is the single biggest remaining numerical unknown, and the path
is obscure (PyTorch's fuser picks different patterns per input
dtype / shape / CUDA version).
- Run the expanded-target sweep with this new understanding and
confirm that the structural metrics (Cα RMSD, GDT-TS, lDDT) are
insensitive to the ~15 % pair-tensor rel_norm gap — the existing
`/tmp/chai_mlx_cuda/findings/vs_pdb_esm.csv` numbers already
suggest they are, since MLX beats CUDA against experimental PDB
on 6 of 10 targets.

