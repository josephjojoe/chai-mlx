"""MLX vs CUDA numerical parity harness.

Consumes a CUDA intermediates NPZ produced by
``modal run -m cuda_harness.run_intermediates`` and replays the MLX pipeline
against those exact inputs, comparing each stage's outputs to the CUDA
captures.

What it checks, in order (each stage feeds CUDA-captured boundary inputs
to MLX so upstream drift doesn't poison downstream comparisons):

1. **Feature embedding.**  MLX ``InputEmbedder`` consumes the raw
   collator features captured from CUDA, and the resulting single /
   pair / atom / msa / template representations are compared against
   the CUDA embedding dumps.
2. **Trunk recycles.**  Starting from the CUDA embedding outputs (not
   the MLX ones, to isolate trunk error), MLX trunk is stepped through
   N recycles and its per-recycle ``(single, pair)`` outputs are
   compared against ``trunk.recycle_<i>.*`` in the NPZ.
3. **Diffusion snapshot.**  For each step that was snapshotted on the
   CUDA side (``diffusion.step_<k>.*``), the MLX denoiser is invoked
   with the CUDA ``atom_pos_hat`` and ``sigma_curr`` and the resulting
   denoised positions are compared against the CUDA denoise output.
4. **Confidence head.**  MLX confidence head is fed CUDA trunk outputs
   plus CUDA final atom positions, and its logits are compared against
   the CUDA logits.

For each comparison we print ``max / mean / p99 / rel_range`` absolute
error. Pass/fail is driven by an optional per-stage tolerance; the
built-in defaults are intended to catch structural mismatches without
being so tight that ordinary backend rounding differences dominate.

Usage
-----

::

    python scripts/cuda_parity.py \\
        --weights-dir weights \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --compute-dtype reference

Precision notes: the reference trunk / token embedder / confidence
head graphs bake in ``torch.autocast("cuda", dtype=torch.bfloat16)``
as explicit ``aten::to`` casts around every linear (bf16) and every
layer_norm / softmax (fp32). The reference diffusion module runs in
pure fp32. See ``cuda_harness.run_intermediates`` for the full
precision write-up.

``--compute-dtype reference`` matches the reference's precision policy
directly for the trunk and confidence stages (that mode uses bf16 there
and keeps diffusion in fp32). ``--compute-dtype float32`` is useful as a
diagnostic (it rules out MLX-side bf16
accumulation as a cause) but is *not* bit-identical against CUDA
even on algorithmic ops: CUDA's scripted module still casts to bf16
inside the graph before every linear, while MLX at float32 does not.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.data.types import (
    EmbeddingOutputs,
    FeatureContext,
    StructureInputs,
    TrunkOutputs,
)
from chai_mlx.model.diffusion import DiffusionModule  # noqa: F401  (sanity import)
from chai_mlx.utils import resolve_dtype


@lru_cache(maxsize=1)
def _require_chai_lab():
    """Import the chai-lab pieces the parity harness needs lazily."""
    try:
        from chai_lab.chai1 import feature_generators
        from chai_lab.data.parsing.structure.entity_type import (
            EntityType as ChaiEntityType,
        )
    except ImportError as exc:
        raise SystemExit(
            "cuda_parity requires chai_lab (for feature_generators + EntityType). "
            "Install via: pip install -e ."
        ) from exc
    return feature_generators, ChaiEntityType


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def _read_manifest(data: dict[str, np.ndarray]) -> dict:
    raw = data.get("_manifest_json")
    if raw is None:
        return {}
    return json.loads(bytes(raw).decode())


def _as_mx(arr: np.ndarray, dtype: mx.Dtype | None = None) -> mx.array:
    out = mx.array(arr)
    if dtype is not None and out.dtype != dtype:
        out = out.astype(dtype)
    return out


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_structure_inputs(data: dict[str, np.ndarray]) -> StructureInputs:
    _, ChaiEntityType = _require_chai_lab()
    b = data["inputs.batch.token_exists_mask"]
    token_exists = b.astype(np.float32)
    atom_exists = data["inputs.batch.atom_exists_mask"].astype(np.float32)
    token_pair_mask = np.einsum("bi,bj->bij", token_exists, token_exists)

    token_entity_type = data["inputs.batch.token_entity_type"].astype(np.int64)
    is_polymer = np.zeros_like(token_entity_type, dtype=np.float32)
    # EntityType: PROTEIN=1, RNA=3, DNA=4 (see chai_lab.data.parsing.structure.entity_type).
    # We use the chai-lab enum here so the mapping stays correct regardless of
    # how the upstream enum is ordered.
    for v in (
        ChaiEntityType.PROTEIN.value,
        ChaiEntityType.RNA.value,
        ChaiEntityType.DNA.value,
    ):
        is_polymer[token_entity_type == v] = 1.0

    q_idx = data["inputs.batch.block_atom_pair_q_idces"]
    kv_idx = data["inputs.batch.block_atom_pair_kv_idces"]
    if q_idx.ndim == 2:
        q_idx = np.broadcast_to(q_idx[None], (token_exists.shape[0], *q_idx.shape)).copy()
    if kv_idx.ndim == 2:
        kv_idx = np.broadcast_to(kv_idx[None], (token_exists.shape[0], *kv_idx.shape)).copy()

    template_mask = data["inputs.batch.template_mask"].astype(np.float32)
    template_input_masks = np.einsum("btn,btm->btnm", template_mask, template_mask)

    return StructureInputs(
        atom_exists_mask=_as_mx(atom_exists),
        token_exists_mask=_as_mx(token_exists),
        token_pair_mask=_as_mx(token_pair_mask),
        atom_token_index=_as_mx(data["inputs.batch.atom_token_index"].astype(np.int64)),
        atom_within_token_index=_as_mx(
            data["inputs.batch.atom_within_token_index"].astype(np.int64)
        ),
        token_reference_atom_index=_as_mx(
            data["inputs.batch.token_ref_atom_index"].astype(np.int64)
        ),
        token_centre_atom_index=_as_mx(
            data["inputs.batch.token_centre_atom_index"].astype(np.int64)
        ),
        token_asym_id=_as_mx(data["inputs.batch.token_asym_id"].astype(np.int64)),
        token_entity_id=_as_mx(data["inputs.batch.token_entity_id"].astype(np.int64)),
        token_chain_id=_as_mx(data["inputs.batch.token_asym_id"].astype(np.int64)),
        token_is_polymer=_as_mx(is_polymer),
        atom_ref_positions=_as_mx(data["inputs.batch.atom_ref_pos"].astype(np.float32)),
        atom_ref_space_uid=_as_mx(data["inputs.batch.atom_ref_space_uid"].astype(np.int64)),
        atom_q_indices=_as_mx(q_idx),
        atom_kv_indices=_as_mx(kv_idx),
        block_atom_pair_mask=_as_mx(data["inputs.batch.block_atom_pair_mask"].astype(np.float32)),
        msa_mask=_as_mx(data["inputs.batch.msa_mask"]),
        template_input_masks=_as_mx(template_input_masks),
        token_residue_index=_as_mx(data["inputs.batch.token_residue_index"].astype(np.int64)),
        token_entity_type=_as_mx(token_entity_type),
        token_backbone_frame_mask=_as_mx(data["inputs.batch.token_backbone_frame_mask"]),
        token_backbone_frame_index=_as_mx(
            data["inputs.batch.token_backbone_frame_index"].astype(np.int64)
        ),
    )


def _reconstruct_feature_context(data: dict[str, np.ndarray]) -> FeatureContext:
    feature_generators, _ = _require_chai_lab()
    structure = _reconstruct_structure_inputs(data)

    raw: dict[str, mx.array] = {}
    for name in feature_generators:
        key = f"inputs.features.{name}"
        if key not in data:
            raise KeyError(
                f"NPZ is missing required feature tensor {key!r}. "
                "Was it produced by run_intermediates.py?"
            )
        raw[name] = _as_mx(data[key])

    bond = _as_mx(data["inputs.bond_ft"])
    empty = mx.zeros((structure.token_exists_mask.shape[0], 0))

    return FeatureContext(
        token_features=empty,
        token_pair_features=empty,
        atom_features=empty,
        atom_pair_features=empty,
        msa_features=empty,
        template_features=empty,
        structure_inputs=structure,
        bond_adjacency=bond,
        raw_features=raw,
    )


def _reconstruct_embedding_outputs(
    data: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    dtype: mx.Dtype,
) -> EmbeddingOutputs:
    """Build an ``EmbeddingOutputs`` dataclass from CUDA-captured boundary tensors.

    We want both halves of the pair representation in their *post-bond-fused*
    form, since the MLX ``InputEmbedder`` fuses bond projection before
    handing off to the trunk / diffusion. Newer NPZs include the fused
    tensors directly; older ones fall back to re-fusing on the fly.
    """
    if "embedding.token_pair_trunk_with_bond" in data:
        token_pair_input = data["embedding.token_pair_trunk_with_bond"].astype(np.float32)
    else:
        token_pair_input = (
            data["embedding.token_pair_trunk"].astype(np.float32)
            + data.get(
                "embedding.bond_trunk",
                np.zeros_like(data["embedding.token_pair_trunk"]),
            ).astype(np.float32)
        )
    if "embedding.token_pair_structure_with_bond" in data:
        token_pair_structure = data["embedding.token_pair_structure_with_bond"].astype(np.float32)
    else:
        token_pair_structure = (
            data["embedding.token_pair_structure"].astype(np.float32)
            + data.get(
                "embedding.bond_structure",
                np.zeros_like(data["embedding.token_pair_structure"]),
            ).astype(np.float32)
        )
    return EmbeddingOutputs(
        token_single_input=_as_mx(data["embedding.token_single"], dtype),
        token_pair_input=_as_mx(token_pair_input, dtype),
        token_pair_structure_input=_as_mx(token_pair_structure, dtype),
        atom_single_input=_as_mx(data["embedding.atom_single_trunk"], dtype),
        atom_single_structure_input=_as_mx(data["embedding.atom_single_structure"], dtype),
        atom_pair_input=_as_mx(data["embedding.atom_pair_trunk"], dtype),
        atom_pair_structure_input=_as_mx(data["embedding.atom_pair_structure"], dtype),
        msa_input=_as_mx(data["embedding.msa"], dtype),
        template_input=_as_mx(data["embedding.templates"], dtype),
        single_initial=_as_mx(data["embedding.token_single_initial"], dtype),
        single_structure=_as_mx(data["embedding.token_single_structure"], dtype),
        pair_initial=_as_mx(data["embedding.token_pair_initial"], dtype),
        pair_structure=_as_mx(token_pair_structure, dtype),
        structure_inputs=structure,
    )


def _reconstruct_trunk_outputs(
    data: dict[str, np.ndarray],
    emb: EmbeddingOutputs,
    *,
    dtype: mx.Dtype,
) -> TrunkOutputs:
    return TrunkOutputs(
        single_initial=emb.single_initial,
        single_trunk=_as_mx(data["trunk.final.single"], dtype),
        single_structure=emb.single_structure,
        pair_initial=emb.pair_initial,
        pair_trunk=_as_mx(data["trunk.final.pair"], dtype),
        pair_structure=emb.pair_structure,
        atom_single_structure_input=emb.atom_single_structure_input,
        atom_pair_structure_input=emb.atom_pair_structure_input,
        msa_input=emb.msa_input,
        template_input=emb.template_input,
        structure_inputs=emb.structure_inputs,
    )


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    name: str
    shape: tuple
    max_abs: float
    mean_abs: float
    p99_abs: float
    ref_range: float
    rel: float
    passed: bool
    tol: float

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"  [{status}] {self.name}: max={self.max_abs:.3e} "
            f"mean={self.mean_abs:.3e} p99={self.p99_abs:.3e} "
            f"rel={self.rel:.4f} ref_range={self.ref_range:.3e} tol={self.tol:.2e}"
        )


def _compare(
    name: str,
    cuda: np.ndarray,
    mlx: np.ndarray,
    tol: float,
    *,
    mask: np.ndarray | None = None,
) -> Comparison:
    """Tensor-for-tensor comparison with robust handling of structural nulls.

    If ``mask`` is provided, only entries where ``mask`` is truthy are
    considered (e.g. non-pad tokens, non-empty MSA rows). The pre-mask
    ``ref_range`` is still reported so callers can spot the "reference is
    uniformly zero" case distinctly from the "reference has values but
    MLX also matched zero" case.

    NaN / Inf in either side are reported as a hard fail with ``max_abs =
    inf`` and a readable ``name`` suffix so the symptom is visible in
    the log rather than poisoning downstream ``rel`` math silently.
    """
    cuda_f = cuda.astype(np.float32)
    mlx_f = mlx.astype(np.float32)
    if cuda_f.shape != mlx_f.shape:
        return Comparison(
            name=f"{name} [shape mismatch: cuda={cuda_f.shape} mlx={mlx_f.shape}]",
            shape=cuda_f.shape,
            max_abs=float("inf"),
            mean_abs=float("inf"),
            p99_abs=float("inf"),
            ref_range=0.0,
            rel=float("inf"),
            passed=False,
            tol=tol,
        )

    cuda_bad = ~np.isfinite(cuda_f)
    mlx_bad = ~np.isfinite(mlx_f)
    if cuda_bad.any() or mlx_bad.any():
        suffix = []
        if cuda_bad.any():
            suffix.append(f"cuda has {int(cuda_bad.sum())} non-finite")
        if mlx_bad.any():
            suffix.append(f"mlx has {int(mlx_bad.sum())} non-finite")
        return Comparison(
            name=f"{name} [{'; '.join(suffix)}]",
            shape=cuda_f.shape,
            max_abs=float("inf"),
            mean_abs=float("inf"),
            p99_abs=float("inf"),
            ref_range=float(np.nanmax(np.abs(cuda_f))) if np.isfinite(cuda_f).any() else 0.0,
            rel=float("inf"),
            passed=False,
            tol=tol,
        )

    # Full-tensor ref_range (always reported pre-mask so "entire tensor is
    # zero" is visually distinct from "masked subset is zero").
    full_ref_range = float(np.abs(cuda_f).max()) if cuda_f.size else 0.0

    if mask is not None:
        m = np.asarray(mask).astype(bool)
        while m.ndim < cuda_f.ndim:
            m = m[..., None]
        m = np.broadcast_to(m, cuda_f.shape)
        if not m.any():
            # Structural null: mask selects zero entries. Compare nothing.
            return Comparison(
                name=f"{name} [mask selects 0 entries]",
                shape=cuda_f.shape,
                max_abs=0.0,
                mean_abs=0.0,
                p99_abs=0.0,
                ref_range=full_ref_range,
                rel=0.0,
                passed=True,
                tol=tol,
            )
        cuda_f = cuda_f[m]
        mlx_f = mlx_f[m]

    diff = np.abs(cuda_f - mlx_f)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    p99_abs = float(np.percentile(diff, 99)) if diff.size else 0.0
    masked_ref_range = float(np.abs(cuda_f).max()) if cuda_f.size else 0.0
    ref_range = masked_ref_range if mask is not None else full_ref_range

    if ref_range == 0.0:
        # Masked-in region of reference is identically zero. The only
        # meaningful question is whether MLX matched.
        rel = 0.0 if max_abs == 0.0 else float("inf")
    else:
        rel = max_abs / ref_range
    return Comparison(
        name=name,
        shape=cuda_f.shape if mask is None else cuda.shape,
        max_abs=max_abs,
        mean_abs=mean_abs,
        p99_abs=p99_abs,
        ref_range=ref_range,
        rel=rel,
        passed=max_abs <= tol,
        tol=tol,
    )


def _tensor_to_numpy(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    return np.array(x, copy=False)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def check_embedding(
    model: ChaiMLX,
    ctx: FeatureContext,
    data: dict[str, np.ndarray],
    tol: float,
) -> list[Comparison]:
    print("\n" + "=" * 70)
    print("  EMBEDDING PARITY: MLX InputEmbedder(raw CUDA features)")
    print("=" * 70)

    emb = model.embed_inputs(ctx)
    mx.eval(
        emb.single_initial,
        emb.pair_initial,
        emb.token_pair_input,
        emb.token_pair_structure_input,
        emb.atom_single_input,
        emb.atom_single_structure_input,
        emb.atom_pair_input,
        emb.atom_pair_structure_input,
        emb.msa_input,
        emb.template_input,
        emb.single_structure,
    )

    comparisons: list[Comparison] = []

    pair_trunk_cuda = (
        data["embedding.token_pair_trunk"].astype(np.float32)
        + data.get("embedding.bond_trunk", np.zeros_like(data["embedding.token_pair_trunk"])).astype(np.float32)
    )
    pair_structure_cuda = (
        data["embedding.token_pair_structure"].astype(np.float32)
        + data.get("embedding.bond_structure", np.zeros_like(data["embedding.token_pair_structure"])).astype(np.float32)
    )

    # Masks for structurally-padded tensors. chai-lab zero-fills the
    # MSA embedding rows where ``msa_mask`` is False, and zero-fills
    # per-template pair positions where ``template_mask`` is False. MLX
    # follows suit, but the comparison is only meaningful on non-pad
    # entries -- otherwise numerical noise in the masked-out region
    # drowns the signal. Pass the relevant mask so ``_compare`` reports
    # on the region that actually participates in downstream compute.
    msa_mask = data.get("inputs.batch.msa_mask")
    template_mask = data.get("inputs.batch.template_mask")

    field_mapping = [
        ("token_single", "token_single_input", data["embedding.token_single"], None),
        ("atom_single_trunk", "atom_single_input", data["embedding.atom_single_trunk"], None),
        ("atom_single_structure", "atom_single_structure_input", data["embedding.atom_single_structure"], None),
        ("atom_pair_trunk", "atom_pair_input", data["embedding.atom_pair_trunk"], None),
        ("atom_pair_structure", "atom_pair_structure_input", data["embedding.atom_pair_structure"], None),
        ("msa", "msa_input", data["embedding.msa"], msa_mask),
        ("templates", "template_input", data["embedding.templates"], template_mask),
        ("token_single_initial", "single_initial", data["embedding.token_single_initial"], None),
        ("token_pair_initial", "pair_initial", data["embedding.token_pair_initial"], None),
    ]
    for label, attr, cuda_arr, mask in field_mapping:
        got = _tensor_to_numpy(getattr(emb, attr))
        c = _compare(label, cuda_arr, got, tol, mask=mask)
        comparisons.append(c)
        print(c.format())

    # Pair/bond-fused tensors must include the bond contribution.
    c = _compare("token_pair_trunk_with_bond", pair_trunk_cuda, _tensor_to_numpy(emb.token_pair_input), tol)
    comparisons.append(c)
    print(c.format())
    c = _compare(
        "token_pair_structure_with_bond",
        pair_structure_cuda,
        _tensor_to_numpy(emb.token_pair_structure_input),
        tol,
    )
    comparisons.append(c)
    print(c.format())
    return comparisons


def check_trunk(
    model: ChaiMLX,
    cuda_emb: EmbeddingOutputs,
    data: dict[str, np.ndarray],
    *,
    num_recycles: int,
    tol: float,
) -> list[Comparison]:
    print("\n" + "=" * 70)
    print("  TRUNK PARITY: MLX Trunk(CUDA embedding outputs)")
    print("=" * 70)

    # Replay trunk recycle-by-recycle so we can compare after each.
    single_init = cuda_emb.single_initial
    pair_init = cuda_emb.pair_initial
    si = cuda_emb.structure_inputs

    prev_single = single_init
    prev_pair = pair_init

    comparisons: list[Comparison] = []
    for rec in range(num_recycles):
        single = single_init + model.trunk_module.token_single_recycle_proj(prev_single)
        pair = pair_init + model.trunk_module.token_pair_recycle_proj(prev_pair)
        mx.eval(single, pair)

        pair = model.trunk_module.template_embedder(
            pair,
            cuda_emb.template_input,
            template_input_masks=si.template_input_masks,
            token_pair_mask=si.token_pair_mask,
        )
        mx.eval(pair)

        pair = model.trunk_module.msa_module(
            single,
            pair,
            cuda_emb.msa_input,
            token_pair_mask=si.token_pair_mask,
            msa_mask=si.msa_mask,
        )
        mx.eval(pair)

        single, pair = model.trunk_module.pairformer_stack(
            single,
            pair,
            pair_mask=si.token_pair_mask,
            single_mask=si.token_exists_mask,
        )
        mx.eval(single, pair)

        single_cuda_key = f"trunk.recycle_{rec}.single"
        pair_cuda_key = f"trunk.recycle_{rec}.pair"
        if single_cuda_key in data:
            c = _compare(
                f"recycle_{rec}.single",
                data[single_cuda_key],
                _tensor_to_numpy(single),
                tol,
            )
            comparisons.append(c)
            print(c.format())
        if pair_cuda_key in data:
            c = _compare(
                f"recycle_{rec}.pair",
                data[pair_cuda_key],
                _tensor_to_numpy(pair),
                tol,
            )
            comparisons.append(c)
            print(c.format())

        prev_single, prev_pair = single, pair

    if "trunk.final.single" in data:
        c = _compare("trunk.final.single", data["trunk.final.single"], _tensor_to_numpy(single), tol)
        comparisons.append(c)
        print(c.format())
    if "trunk.final.pair" in data:
        c = _compare("trunk.final.pair", data["trunk.final.pair"], _tensor_to_numpy(pair), tol)
        comparisons.append(c)
        print(c.format())

    return comparisons


def check_diffusion_snapshots(
    model: ChaiMLX,
    cuda_emb: EmbeddingOutputs,
    data: dict[str, np.ndarray],
    *,
    dtype: mx.Dtype,
    tol: float,
) -> list[Comparison]:
    print("\n" + "=" * 70)
    print("  DIFFUSION SNAPSHOT PARITY: MLX denoise(CUDA trunk + CUDA atom_pos_hat)")
    print("=" * 70)

    trunk_out = _reconstruct_trunk_outputs(data, cuda_emb, dtype=dtype)
    cache = model.prepare_diffusion_cache(trunk_out)
    mx.eval(
        cache.s_static,
        cache.z_cond,
        cache.blocked_pair_base,
        cache.atom_cond,
        cache.atom_single_cond,
        *cache.pair_biases,
    )

    step_keys = sorted(
        {
            int(k.split("_")[1].split(".")[0])
            for k in data
            if k.startswith("diffusion.step_") and k.endswith(".denoised")
        }
    )
    comparisons: list[Comparison] = []
    for step_idx in step_keys:
        tag = f"diffusion.step_{step_idx:04d}"
        atom_pos_hat = _as_mx(data[f"{tag}.atom_pos_hat"], dtype)
        sigma_curr = float(data[f"{tag}.sigma_curr"])
        # Expand sigma to [1, ds] and atom_pos_hat to [1, ds, A, 3].
        if atom_pos_hat.ndim == 3:
            ds = atom_pos_hat.shape[0]
            atom_pos_hat_bds = atom_pos_hat.reshape(1, ds, atom_pos_hat.shape[1], atom_pos_hat.shape[2])
        else:
            ds = atom_pos_hat.shape[1]
            atom_pos_hat_bds = atom_pos_hat
        sigma_mx = mx.full((1, ds), sigma_curr, dtype=mx.float32)

        denoised = model.diffusion_module.denoise(cache, atom_pos_hat_bds, sigma_mx)
        mx.eval(denoised)

        cuda_denoised = data[f"{tag}.denoised"].astype(np.float32)
        # CUDA records denoised_pos as [1, ds, A, 3] already; MLX returns the same shape.
        got = _tensor_to_numpy(denoised)
        if got.shape != cuda_denoised.shape and got.ndim == cuda_denoised.ndim + 1:
            got = got[0]
        elif got.ndim == cuda_denoised.ndim - 1:
            got = got[None]
        c = _compare(f"{tag}.denoised", cuda_denoised, got, tol)
        comparisons.append(c)
        print(c.format())

    return comparisons


def check_confidence(
    model: ChaiMLX,
    cuda_emb: EmbeddingOutputs,
    data: dict[str, np.ndarray],
    *,
    dtype: mx.Dtype,
    tol: float,
) -> list[Comparison]:
    print("\n" + "=" * 70)
    print("  CONFIDENCE PARITY: MLX confidence(CUDA trunk + CUDA final coords)")
    print("=" * 70)

    trunk_out = _reconstruct_trunk_outputs(data, cuda_emb, dtype=dtype)
    final_coords = _as_mx(data["diffusion.atom_pos_final"], mx.float32)
    # CUDA captures atom_pos as [ds, A, 3]; MLX's confidence head accepts
    # either [b, a, 3] or [b, ds, a, 3]. Promote to [1, ds, A, 3] so we
    # match the ds-aware branch.
    if final_coords.ndim == 3:
        final_coords = final_coords[None]
    conf = model.confidence(trunk_out, final_coords)
    mx.eval(conf.pae_logits, conf.pde_logits, conf.plddt_logits)

    comparisons: list[Comparison] = []
    for attr, key in (
        ("pae_logits", "confidence.pae_logits"),
        ("pde_logits", "confidence.pde_logits"),
        ("plddt_logits", "confidence.plddt_logits"),
    ):
        got = _tensor_to_numpy(getattr(conf, attr))
        cuda_arr = data[key]
        # CUDA concatenates per-sample logits along axis 0 -> shape [ds, ...].
        # MLX emits [b=1, ds, ...] when called with 4D coords. Strip the
        # leading batch axis so shapes align before comparing.
        if got.ndim == cuda_arr.ndim + 1 and got.shape[0] == 1:
            got = got[0]
        c = _compare(key, cuda_arr, got, tol)
        comparisons.append(c)
        print(c.format())
    return comparisons


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _default_tolerance(stage: str, dtype: str) -> float:
    # These tolerances are designed to flag structural mismatches; they are
    # *not* so tight that bf16 fused-kernel rounding differences between MLX
    # and CUDA will fail them.
    if dtype == "float32":
        return {
            "embedding": 1e-3,
            "trunk": 5e-2,
            "diffusion": 5e-2,
            "confidence": 1e-2,
        }[stage]
    return {
        "embedding": 1e-2,
        "trunk": 5e0,
        "diffusion": 5e-1,
        "confidence": 1e0,
    }[stage]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True, help="intermediates bundle from run_intermediates.py")
    parser.add_argument("--compute-dtype", default=None, choices=["reference", "float32"])
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-trunk", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--skip-confidence", action="store_true")
    parser.add_argument("--tol-embedding", type=float, default=None)
    parser.add_argument("--tol-trunk", type=float, default=None)
    parser.add_argument("--tol-diffusion", type=float, default=None)
    parser.add_argument("--tol-confidence", type=float, default=None)
    parser.add_argument("--summary-json", type=Path, default=None, help="optional path to dump pass/fail summary")
    args = parser.parse_args(list(argv) if argv is not None else None)

    _require_chai_lab()

    print(f"[load] {args.npz}")
    data = _load_npz(args.npz)
    manifest = _read_manifest(data)
    if manifest:
        print(
            f"  target={manifest.get('target')} seed={manifest.get('seed')} "
            f"n_tokens={manifest.get('n_tokens')} gpu={manifest.get('gpu_name')}"
        )

    print(f"[load] model weights from {args.weights_dir}")
    model = ChaiMLX.from_pretrained(
        args.weights_dir, strict=False, compute_dtype=args.compute_dtype
    )
    dtype = resolve_dtype(model.cfg)
    dtype_name = "float32" if dtype == mx.float32 else "reference"
    print(f"  compute_dtype={dtype_name}")

    tols = {
        "embedding": args.tol_embedding or _default_tolerance("embedding", dtype_name),
        "trunk": args.tol_trunk or _default_tolerance("trunk", dtype_name),
        "diffusion": args.tol_diffusion or _default_tolerance("diffusion", dtype_name),
        "confidence": args.tol_confidence or _default_tolerance("confidence", dtype_name),
    }

    ctx = _reconstruct_feature_context(data)
    cuda_emb = _reconstruct_embedding_outputs(data, ctx.structure_inputs, dtype=dtype)

    summary: dict[str, list[dict]] = {}

    if not args.skip_embedding:
        comps = check_embedding(model, ctx, data, tol=tols["embedding"])
        summary["embedding"] = [c.__dict__ for c in comps]
        gc.collect()
        mx.clear_cache()

    num_recycles = int(manifest.get("num_recycles", 3))
    if not args.skip_trunk:
        comps = check_trunk(
            model, cuda_emb, data, num_recycles=num_recycles, tol=tols["trunk"]
        )
        summary["trunk"] = [c.__dict__ for c in comps]
        gc.collect()
        mx.clear_cache()

    if not args.skip_diffusion:
        comps = check_diffusion_snapshots(
            model, cuda_emb, data, dtype=dtype, tol=tols["diffusion"]
        )
        summary["diffusion"] = [c.__dict__ for c in comps]
        gc.collect()
        mx.clear_cache()

    if not args.skip_confidence:
        comps = check_confidence(
            model, cuda_emb, data, dtype=dtype, tol=tols["confidence"]
        )
        summary["confidence"] = [c.__dict__ for c in comps]

    total = sum(len(v) for v in summary.values())
    failed = sum(1 for v in summary.values() for c in v if not c["passed"])
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {total - failed}/{total} passed ({failed} failed)")
    print("=" * 70)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_out = {
            "npz": str(args.npz),
            "weights_dir": str(args.weights_dir),
            "compute_dtype": dtype_name,
            "manifest": manifest,
            "tolerances": tols,
            "comparisons": summary,
            "totals": {"total": total, "failed": failed},
        }
        args.summary_json.write_text(json.dumps(summary_out, indent=2, default=str))
        print(f"[save] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
