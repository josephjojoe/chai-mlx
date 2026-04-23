"""End-to-end Chai-MLX inference runner for arbitrary FASTA inputs.

This is the canonical implementation of the ``chai-mlx-infer`` console
script declared in ``pyproject.toml``. It exposes the full
:func:`chai_mlx.data.featurize.featurize_fasta` surface (constraints,
offline MSA, online MSA server, offline templates, online templates
server, and all four ESM backends), runs the production inference
pipeline, and writes per-sample CIFs, scores, and a run manifest.

Output layout under ``--output-dir``::

    <output-dir>/
        input.fasta                 copy of the input FASTA (for reference)
        pred.model_idx_0.cif         one CIF per diffusion sample
        pred.model_idx_1.cif
        ...
        scores.json                  aggregate_score / ptm / iptm per sample
        scores.model_idx_{0..N}.npz  per-sample npz (chai-lab parity)
        manifest.json                dtype, recycles, steps, wall clock, etc.
        coords.npz                   optional: raw coords + scores (--save-npz)
        msa_coverage.png             optional: MSA depth plot (--write-msa-plot)

In batch mode (``--fasta-dir``) the top level holds one subdirectory
per FASTA with the above layout, plus a ``run_summary.json`` listing
per-file status and wall clock.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize_fasta


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chai-MLX end-to-end inference for arbitrary FASTA inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core = parser.add_argument_group("core inputs")
    core.add_argument("--weights-dir", type=Path, required=True,
                      help="Directory containing config.json + safetensors, or a HF repo id.")
    core.add_argument("--fasta", type=Path, default=None,
                      help="Input FASTA file. Header format: '>kind|name=SHORT' where "
                           "kind ∈ {protein, ligand, dna, rna, glycan} and SHORT ≤ 4 chars. "
                           "Mutually exclusive with --fasta-dir; exactly one must be given.")
    core.add_argument("--fasta-dir", type=Path, default=None,
                      help="Directory containing multiple '*.fasta' files. The model is "
                           "loaded once and each FASTA is folded into "
                           "<output-dir>/<fasta_stem>/ with the same flags as a single "
                           "--fasta run. Per-file validation errors are logged and the "
                           "next file is attempted; a run_summary.json in --output-dir "
                           "records per-file status + wall clock. Mutually exclusive "
                           "with --fasta.")
    core.add_argument("--output-dir", type=Path, required=True,
                      help="Directory for CIFs, scores.json, manifest.json (and optional "
                           "npz). In batch mode (--fasta-dir), each FASTA goes into "
                           "<output-dir>/<fasta_stem>/ and a run_summary.json is written "
                           "at the top level.")

    model = parser.add_argument_group("model / sampling")
    model.add_argument(
        "--dtype",
        default="reference",
        choices=["reference", "float32"],
        help='Precision policy: "reference" matches the reference bundle '
             "(bf16 trunk/confidence, fp32 diffusion); "
             '"float32" keeps the MLX port in fp32.',
    )
    model.add_argument("--recycles", type=int, default=3,
                       help="Number of trunk recycles (chai-1 'num_trunk_recycles').")
    model.add_argument("--num-steps", type=int, default=200,
                       help="Diffusion timesteps (chai-1 'num_diffn_timesteps').")
    model.add_argument("--num-samples", type=int, default=5,
                       help="Diffusion samples per trunk sample (chai-1 "
                            "'num_diffn_samples').")
    model.add_argument("--num-trunk-samples", type=int, default=1,
                       help="Independent trunk runs at seeds "
                            "{seed, seed+1, ..., seed+N-1} (chai-1 "
                            "'num_trunk_samples'). Each trunk sample spawns "
                            "--num-samples diffusion samples, so the total "
                            "number of candidate CIFs is "
                            "--num-trunk-samples * --num-samples. When this "
                            "is > 1, outputs land under "
                            "<output-dir>/trunk_<i>/; otherwise the layout "
                            "is flat (matching chai-lab's convention).")
    model.add_argument("--recycle-msa-subsample", type=int, default=0,
                       help="If > 0, subsample the MSA differently each "
                            "trunk recycle to decorrelate samples (chai-1 "
                            "'recycle_msa_subsample'). 0 (default) disables "
                            "subsampling and uses the full MSA every recycle.")
    model.add_argument("--seed", type=int, default=42)
    model.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False,
                       help="Use the debug inference path and retain full intermediates.")
    model.add_argument(
        "--pad-strategy",
        choices=["exact", "bucket"],
        default="exact",
        help="How to pad the token/atom axes before inference. "
             "'exact' (default) pads to num_tokens verbatim with n_atoms "
             "rounded up to the next multiple of 32 -- the tightest shape "
             "the MLX kernels accept. 'bucket' pads up to the smallest of "
             "chai-lab's seven reference crop sizes "
             "[256, 384, 512, 768, 1024, 1536, 2048] with n_atoms = 23 * "
             "n_tokens; use this when you want those same traced bucket sizes.",
    )

    constraints = parser.add_argument_group("constraints")
    constraints.add_argument("--constraint-path", type=Path, default=None,
                             help="Chai-lab constraint CSV (contact + pocket + covalent-bond "
                                  "restraints).")

    msa = parser.add_argument_group("MSA (mutually exclusive modes)")
    msa.add_argument("--msa-directory", type=Path, default=None,
                     help="Pre-computed MSA directory of chai-lab '.aligned.pqt' "
                          "parquet files (filenames from "
                          "chai_lab.data.parsing.msas.aligned_pqt.expected_basename). "
                          "The easiest way to produce one is to run once with "
                          "--use-msa-server and reuse the resulting "
                          "<feature-dir>/mlx_features/msas directory here.")
    msa.add_argument("--use-msa-server", action=argparse.BooleanOptionalAction, default=False,
                     help="Fetch MSAs from the free public ColabFold API "
                          "(https://api.colabfold.com). Mutually exclusive with "
                          "--msa-directory. If an msa directory from a prior run "
                          "already exists at <feature-dir>/mlx_features/msas, it is "
                          "reused automatically; pass --refresh-msa to force a "
                          "fresh fetch.")
    msa.add_argument("--msa-server-url", type=str, default="https://api.colabfold.com",
                     help="ColabFold MSA server endpoint.")
    msa.add_argument("--refresh-msa", action=argparse.BooleanOptionalAction, default=False,
                     help="With --use-msa-server, delete any existing "
                          "<feature-dir>/mlx_features/msas directory before fetching "
                          "so the server is always queried fresh.")

    templates = parser.add_argument_group("templates (mutually exclusive modes)")
    templates.add_argument("--templates-path", type=Path, default=None,
                           help="Pre-computed templates m8 file. Mutually exclusive with "
                                "--use-templates-server. (Plumbed, not validated here.)")
    templates.add_argument("--use-templates-server", action=argparse.BooleanOptionalAction,
                           default=False,
                           help="Fetch templates online (via the MSA server). Requires "
                                "--use-msa-server. (Plumbed, not validated here.)")

    esm = parser.add_argument_group("ESM-2 embeddings")
    esm.add_argument("--esm-backend", choices=["off", "chai", "mlx", "mlx_cache"],
                     default="off",
                     help="'off' (default): zero-fill ESM features. 'chai': chai-lab's "
                          "traced CUDA fp16 checkpoint (CUDA host only). 'mlx': esm-mlx "
                          "in-process (16 GB+ RAM, not recommended on 16 GB Macs). "
                          "'mlx_cache': load pre-computed embeddings from --esm-cache-dir.")
    esm.add_argument("--esm-cache-dir", type=Path, default=None,
                     help="Directory of pre-computed <sha1(seq)>.npy files (produced by "
                           "chai-mlx-precompute-esm). "
                          "Required with --esm-backend mlx_cache.")

    output = parser.add_argument_group("output")
    output.add_argument("--save-npz", type=Path, default=None,
                        help="Optional path to also dump raw coords + scores as .npz "
                             "(CIFs are always produced).")
    output.add_argument("--skip-cif", action=argparse.BooleanOptionalAction, default=False,
                        help="Skip CIF output entirely (scores + optional npz only).")
    output.add_argument("--feature-dir", type=Path, default=None,
                        help="Where chai-lab's featurizer may write intermediate artifacts "
                             "(MSA/template caches, etc.). Defaults to <output-dir>/_features.")
    output.add_argument("--fasta-chain-names",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Use FASTA entity names as CIF chain IDs (chai-1's "
                             "'fasta_names_as_cif_chains' flag). When unset (the "
                             "default), the flag is auto-enabled iff --constraint-path "
                             "is provided, because restraint CSVs address chains by "
                             "FASTA name; otherwise chai-1's upstream default of "
                             "sequential A/B/C... chain IDs is used. Pass the explicit "
                             "flag to force one behaviour or the other.")
    output.add_argument("--write-msa-plot", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Write an MSA coverage plot to "
                             "<output-dir>/msa_coverage.png. Only meaningful when an "
                             "MSA source is active (--use-msa-server or "
                             "--msa-directory); silently skipped otherwise and when "
                             "matplotlib is not installed.")

    args = parser.parse_args(argv)

    if args.esm_backend == "mlx_cache" and args.esm_cache_dir is None:
        parser.error("--esm-backend mlx_cache requires --esm-cache-dir")
    if args.use_msa_server and args.msa_directory is not None:
        parser.error("--use-msa-server and --msa-directory are mutually exclusive")
    if args.use_templates_server and args.templates_path is not None:
        parser.error("--use-templates-server and --templates-path are mutually exclusive")
    if args.use_templates_server and not args.use_msa_server:
        parser.error("--use-templates-server requires --use-msa-server (chai-lab's "
                     "templates-server path is driven by the MSA server pipeline)")

    # Exactly-one-of --fasta / --fasta-dir
    if (args.fasta is None) == (args.fasta_dir is None):
        parser.error(
            "exactly one of --fasta or --fasta-dir must be provided"
        )
    if args.fasta is not None and not args.fasta.exists():
        parser.error(f"--fasta {args.fasta} does not exist")
    if args.fasta_dir is not None:
        if not args.fasta_dir.is_dir():
            parser.error(f"--fasta-dir {args.fasta_dir} is not a directory")
        fastas = sorted(args.fasta_dir.glob("*.fasta"))
        if not fastas:
            parser.error(
                f"--fasta-dir {args.fasta_dir} contains no *.fasta files"
            )

    return args


def _save_cifs(
    *,
    coords_np: "Any",
    per_atom_plddt_np: "Any | None",
    output_dir: Path,
    fasta_path: Path,
    feature_dir: Path,
    entity_name_as_subchain: bool,
    pad_strategy: str = "exact",
) -> list[Path]:
    """Write one chai-lab-format CIF per diffusion sample.

    We rebuild a minimal reference ``chai-lab`` feature context so the
    per-chain asym labels match what :func:`featurize_fasta` produced
    for the main fold. ``entity_name_as_subchain`` must match the
    value passed to ``featurize_fasta`` -- pass False and chai-lab
    emits A/B/C... CIF chain IDs; pass True and it uses the FASTA
    entity names. The reference context is only used to drive
    ``save_to_cif``'s atom/residue bookkeeping, which depends solely
    on the structural layout of the FASTA -- not on MSAs, templates,
    or constraint restraints. So we skip those knobs for the ref
    pass, which keeps this path fast and immune to the caller's
    MSA / template configuration.

    ``pad_strategy`` MUST match the value passed to
    :func:`featurize_fasta` for the same run -- the MLX coords tensor
    has ``A = pad_strategy_padded_n_atoms`` and ``save_to_cif`` walks
    ``output_batch['atom_exists_mask']`` of the same length. Mismatched
    strategies produce either an ``IndexError`` or CIFs with nonsense
    atom indices.

    If ``per_atom_plddt_np`` is provided (shape ``(B, S, A)`` in [0, 1]),
    we pass ``bfactors=per_atom_plddt * 100`` through to
    ``chai_lab.data.io.cif_utils.save_to_cif`` so viewers like PyMOL /
    ChimeraX colour the output by confidence (matching chai-lab's own
    ``chai_lab.chai1.run_inference`` behaviour).
    """

    import torch

    # Install chai-lab's RDKit-timeout workaround before importing its
    # featurizer, so ligand targets do not hit the macOS closure-pickle
    # failure in chai-lab's timeout wrapper.
    from chai_mlx.data._rdkit_timeout_patch import apply_rdkit_timeout_patch
    apply_rdkit_timeout_patch()

    from chai_lab.chai1 import Collate, feature_factory, make_all_atom_feature_context
    from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif
    from chai_mlx.data.featurize import _override_pad_strategy

    ref_feature_dir = feature_dir / "ref_features"
    ref_ctx = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=ref_feature_dir,
        entity_name_as_subchain=entity_name_as_subchain,
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        esm_device=torch.device("cpu"),
    )
    collator = Collate(feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32)
    with _override_pad_strategy(pad_strategy):
        output_batch = collator([ref_ctx])["inputs"]
    asym_entity_names = {
        i: get_chain_letter(i) for i, _ in enumerate(ref_ctx.chains, start=1)
    }

    # ``coords_np`` has shape ``(B, S, A, 3)``.  We take B=0 and write one
    # CIF per sample.  ``save_to_cif`` expects ``(1, A, 3)`` tensors.
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = coords_np.shape[1]
    cif_paths: list[Path] = []
    for s in range(n_samples):
        cif_path = output_dir / f"pred.model_idx_{s}.cif"
        bfactors = None
        if per_atom_plddt_np is not None:
            # save_to_cif wants Float[Tensor, '1 n_atoms']; per_atom_plddt
            # is (B, S, A) in [0, 1] -- scale to 0-100 to match chai-lab
            # output (see chai-lab/chai_lab/chai1.py:1028).
            bfactors = torch.from_numpy(
                per_atom_plddt_np[:, s].astype("float32") * 100.0
            )
        save_to_cif(
            coords=torch.from_numpy(coords_np[:, s]),
            bfactors=bfactors,
            output_batch=output_batch,
            write_path=cif_path,
            asym_entity_names=asym_entity_names,
        )
        cif_paths.append(cif_path)
    return cif_paths


def _scores_to_dict(ranking: "Any") -> dict[str, list[float]]:
    """Flatten MLX ranking outputs to JSON-serialisable per-sample lists."""
    import numpy as np

    def _as_list(arr: "Any") -> list[float]:
        return np.array(arr.astype(mx.float32)).reshape(-1).tolist()

    out: dict[str, list[float]] = {
        "aggregate_score": _as_list(ranking.aggregate_score),
        "ptm": _as_list(ranking.ptm),
        "iptm": _as_list(ranking.iptm),
        "has_inter_chain_clashes": _as_list(ranking.has_inter_chain_clashes),
    }
    if ranking.complex_plddt is not None:
        out["complex_plddt"] = _as_list(ranking.complex_plddt)
    if ranking.total_clashes is not None:
        out["total_clashes"] = _as_list(ranking.total_clashes)
    if ranking.total_inter_chain_clashes is not None:
        out["total_inter_chain_clashes"] = _as_list(ranking.total_inter_chain_clashes)
    return out


def _bin_centers_np(min_bin: float, max_bin: float, no_bins: int):
    """NumPy port of :func:`chai_lab.chai1._bin_centers`.

    Mirrors upstream so decoded PAE/PDE/pLDDT tensors use bit-identical
    bin midpoints; that keeps the emitted ``pae``/``pde``/``plddt``
    arrays aligned with chai-lab's decoded outputs.
    """
    import numpy as np
    return np.linspace(min_bin, max_bin, 2 * no_bins + 1)[1::2]


def _decode_per_token_scores(
    *,
    confidence: "Any",
    structure: "Any",
    num_samples: int,
) -> tuple["Any", "Any", "Any"]:
    """Decode PAE / PDE / per-token pLDDT from the confidence logits.

    Mirrors :func:`chai_lab.chai1.run_folding_on_context`'s
    ``softmax_einsum_and_cpu`` decode (chai1.py lines ~920-958):

        pae[..., i, j]  = Σ_b softmax(pae_logits)_{...ij b} · bin_centers(0, 32, 64)_b
        pde[..., i, j]  = Σ_b softmax(pde_logits)_{...ij b} · bin_centers(0, 32, 64)_b
        plddt_atom[...] = Σ_b softmax(plddt_logits)_{...b} · bin_centers(0, 1, 50)_b
        plddt_token[n]  = mean(plddt_atom[atom_token_index == n])

    Returns per-sample numpy arrays with shapes:

    * ``pae``:   ``(num_samples, n_tokens, n_tokens)`` (masked to live tokens)
    * ``pde``:   ``(num_samples, n_tokens, n_tokens)`` (masked to live tokens)
    * ``plddt``: ``(num_samples, n_tokens)`` (atom-averaged, masked atoms only)

    Token-pair arrays are already masked exactly like
    ``chai_lab.chai1::StructureCandidates.pae``/``pde``: entries for
    padded tokens are zeroed because we slice by ``token_exists_mask``
    before computing, then zero-pad back to the full crop so downstream
    tools can index by token position.
    """
    import numpy as np

    def _softmax_expect(logits_np, bin_centers_np):
        logits_np = logits_np.astype(np.float32, copy=False)
        x = logits_np - logits_np.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return np.einsum("...d,d->...", probs, bin_centers_np.astype(np.float32))

    pae_logits = np.array(confidence.pae_logits.astype(mx.float32))
    pde_logits = np.array(confidence.pde_logits.astype(mx.float32))
    plddt_logits = np.array(confidence.plddt_logits.astype(mx.float32))

    pae_centers = _bin_centers_np(0.0, 32.0, 64)
    pde_centers = _bin_centers_np(0.0, 32.0, 64)
    plddt_centers = _bin_centers_np(0.0, 1.0, plddt_logits.shape[-1])

    pae = _softmax_expect(pae_logits, pae_centers)
    pde = _softmax_expect(pde_logits, pde_centers)
    plddt_per_atom = _softmax_expect(plddt_logits, plddt_centers)

    token_exists = np.array(
        structure.token_exists_mask.astype(mx.float32)
    ).astype(bool)
    atom_mask = np.array(
        structure.atom_exists_mask.astype(mx.float32)
    ).astype(bool)
    atom_token_index = np.array(
        structure.atom_token_index.astype(mx.int32)
    )

    def _pick_sample(arr, s: int):
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == num_samples:
            return arr[0, s]
        if arr.ndim >= 1 and arr.shape[0] == num_samples:
            return arr[s]
        return arr

    def _pick_batch(arr):
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr

    token_exists_1d = _pick_batch(token_exists)
    atom_mask_1d = _pick_batch(atom_mask)
    atom_token_index_1d = _pick_batch(atom_token_index)

    n_tokens = token_exists_1d.shape[0]

    pae_per_sample = np.zeros((num_samples, n_tokens, n_tokens), dtype=np.float32)
    pde_per_sample = np.zeros((num_samples, n_tokens, n_tokens), dtype=np.float32)
    plddt_per_sample = np.zeros((num_samples, n_tokens), dtype=np.float32)

    live_token_idx = np.where(token_exists_1d)[0]
    # Use a 2-D boolean outer mask so the broadcast assignment below
    # matches chai-lab's "zero-pad at padded positions" contract. Rows
    # i and j are "live" iff both i and j exist; everything else stays
    # at the initial 0 we allocated.
    live_pair_mask = np.outer(token_exists_1d, token_exists_1d)

    for s in range(num_samples):
        pae_s = _pick_sample(pae, s)  # (n_tokens, n_tokens)
        pde_s = _pick_sample(pde, s)
        plddt_atom_s = _pick_sample(plddt_per_atom, s)  # (n_atoms,)

        # chai-lab masks pae/pde to live tokens only; we emit full-crop
        # arrays with 0s for padded positions so downstream consumers
        # can index by token position without a mask.
        if live_token_idx.size > 0:
            pae_per_sample[s] = np.where(live_pair_mask, pae_s, 0.0)
            pde_per_sample[s] = np.where(live_pair_mask, pde_s, 0.0)

        # Per-token pLDDT: average per-atom pLDDT over atoms whose
        # atom_token_index == n (masked atoms only), matching
        # chai-lab/chai_lab/chai1.py::avg_per_token_1d.
        live_atom = atom_mask_1d
        if live_atom.any():
            weights = np.where(live_atom, plddt_atom_s, 0.0)
            numer = np.bincount(
                atom_token_index_1d[live_atom],
                weights=weights[live_atom],
                minlength=n_tokens,
            )
            denom = np.bincount(
                atom_token_index_1d[live_atom],
                minlength=n_tokens,
            ).clip(min=1)
            plddt_per_sample[s] = numer / denom

    return pae_per_sample, pde_per_sample, plddt_per_sample


def _write_per_sample_scores(
    *,
    output_dir: Path,
    ranking: "Any",
    confidence: "Any",
    structure: "Any",
    num_samples: int,
) -> list[str]:
    """Write one ``scores.model_idx_{s}.npz`` per diffusion sample.

    Field names mirror :func:`chai_lab.ranking.rank.get_scores` plus
    the per-token tensors carried by
    :class:`chai_lab.chai1.StructureCandidates` (``pae``, ``pde``,
    ``plddt``). The arrays inside each npz are per-sample slices of the
    batched MLX outputs so downstream tools that expect chai-lab's npz
    layout do not need to re-index.

    Per-sample npz schema:

    * ``aggregate_score``, ``ptm``, ``iptm``,
      ``has_inter_chain_clashes`` — chai-lab ``get_scores`` baseline.
    * ``per_chain_ptm``, ``per_chain_pair_iptm``,
      ``chain_chain_clashes`` — chai-lab per-chain breakdowns.
    * ``complex_plddt``, ``per_chain_plddt``, ``per_atom_plddt``,
      ``total_clashes``, ``total_inter_chain_clashes`` — MLX
      supersets (chai-lab does not emit these scalars directly, but
      downstream readers that only look up the shared keys still work).
      ``per_atom_plddt`` is zero at padding-atom slots (live slots
      carry the per-atom pLDDT expectation).
    * ``pae`` ``(n_tokens, n_tokens)``, ``pde`` ``(n_tokens,
      n_tokens)``, ``plddt`` ``(n_tokens,)`` — per-token tensors that
      match :class:`chai_lab.chai1.StructureCandidates`'s ``pae`` /
      ``pde`` / ``plddt`` fields bit-for-bit up to MLX numerical
      noise. Padded token positions are zero-filled.
    """
    import numpy as np

    def _to_np(arr):
        return np.array(arr.astype(mx.float32)) if arr is not None else None

    agg = _to_np(ranking.aggregate_score)          # (B, S) or (S,)
    ptm = _to_np(ranking.ptm)
    iptm = _to_np(ranking.iptm)
    per_chain_ptm = _to_np(ranking.per_chain_ptm)
    per_chain_pair_iptm = _to_np(ranking.per_chain_pair_iptm)
    has_inter = _to_np(ranking.has_inter_chain_clashes)
    chain_chain_clashes = (
        np.array(ranking.chain_chain_clashes.astype(mx.int32))
        if ranking.chain_chain_clashes is not None
        else None
    )
    complex_plddt = _to_np(ranking.complex_plddt)
    per_chain_plddt = _to_np(ranking.per_chain_plddt)
    per_atom_plddt = _to_np(ranking.per_atom_plddt)
    # ``RankingOutputs.per_atom_plddt`` carries an unmasked per-bin
    # expectation over the full ``(..., A_padded)`` tensor.  In bucket
    # mode the padded slots hold softmax expectations over logits the
    # model never trained for, which contaminates any downstream
    # ``.mean()`` / residue-averaging that callers run on this array.
    # Zero-mask the padding slots so ``per_atom_plddt`` is safe to
    # average, sum, or plot without extra masking, matching the
    # zero-padding convention we already use for ``pae`` / ``pde`` /
    # ``plddt`` per-token.  In exact mode this is a no-op since
    # every slot in the tensor is already live.
    if per_atom_plddt is not None:
        atom_mask_np = np.array(
            structure.atom_exists_mask.astype(mx.float32)
        ).astype(bool)
        if atom_mask_np.ndim >= 1 and atom_mask_np.shape[0] == 1:
            atom_mask_np = atom_mask_np[0]
        # Broadcast the atom mask across leading batch/sample axes.
        broadcast_shape = per_atom_plddt.shape[:-1] + atom_mask_np.shape
        atom_mask_b = np.broadcast_to(atom_mask_np, broadcast_shape)
        per_atom_plddt = np.where(atom_mask_b, per_atom_plddt, 0.0)
    total_clashes = (
        np.array(ranking.total_clashes.astype(mx.int32))
        if ranking.total_clashes is not None
        else None
    )
    total_inter = (
        np.array(ranking.total_inter_chain_clashes.astype(mx.int32))
        if ranking.total_inter_chain_clashes is not None
        else None
    )

    pae_per_sample, pde_per_sample, plddt_per_sample = _decode_per_token_scores(
        confidence=confidence,
        structure=structure,
        num_samples=num_samples,
    )

    def _pick_sample(arr, s: int):
        """Return the s-th sample along the canonical sample axis.

        MLX outputs are ``(B=1, S, ...)`` when there is more than one
        sample, or ``(B=1, ...)`` when S was squeezed (Ranker at
        ``coords.ndim == 3``). ``axis=1`` is the sample axis when it
        exists; fall back to the 0-axis when not.
        """
        if arr is None:
            return None
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == num_samples:
            return arr[0, s]
        if arr.ndim >= 1 and arr.shape[0] == num_samples:
            return arr[s]
        # degenerate single-sample tensors: broadcast
        return arr

    paths: list[str] = []
    for s in range(num_samples):
        payload: dict = {
            "aggregate_score": _pick_sample(agg, s),
            "ptm": _pick_sample(ptm, s),
            "iptm": _pick_sample(iptm, s),
            "has_inter_chain_clashes": _pick_sample(has_inter, s),
            # Per-token tensors (chai-lab StructureCandidates parity)
            "pae": pae_per_sample[s],
            "pde": pde_per_sample[s],
            "plddt": plddt_per_sample[s],
        }
        for key, arr in (
            ("per_chain_ptm", per_chain_ptm),
            ("per_chain_pair_iptm", per_chain_pair_iptm),
            ("chain_chain_clashes", chain_chain_clashes),
            ("complex_plddt", complex_plddt),
            ("per_chain_plddt", per_chain_plddt),
            ("per_atom_plddt", per_atom_plddt),
            ("total_clashes", total_clashes),
            ("total_inter_chain_clashes", total_inter),
        ):
            picked = _pick_sample(arr, s)
            if picked is not None:
                payload[key] = picked

        out_path = output_dir / f"scores.model_idx_{s}.npz"
        np.savez(out_path, **payload)
        paths.append(str(out_path))

    return paths


def _fold_one_fasta(
    *,
    model: "ChaiMLX",
    args: argparse.Namespace,
    fasta_path: Path,
    output_dir: Path,
    feature_dir: Path,
    weights_load_seconds: float,
) -> dict:
    """Run a single FASTA through the pipeline. Returns a manifest dict.

    The caller is responsible for FASTA validation (so batch mode can
    surface a full per-file status) and for directory creation.
    """
    fasta_copy = output_dir / "input.fasta"
    try:
        shutil.copyfile(fasta_path, fasta_copy)
    except shutil.SameFileError:
        pass

    # ``chai_lab.chai1.make_all_atom_feature_context`` does
    # ``(output_dir/'msas').mkdir(parents=True, exist_ok=False)`` whenever
    # ``use_msa_server=True``, so a second ``--use-msa-server`` run
    # against an existing --output-dir crashes. Detect and reuse a prior
    # fetch automatically unless --refresh-msa says otherwise; this also
    # makes the common "fetch once, iterate many" workflow painless.
    effective_use_msa_server = args.use_msa_server
    effective_msa_directory = args.msa_directory
    effective_use_templates_server = args.use_templates_server
    effective_templates_path = args.templates_path
    cached_msa_dir = feature_dir / "mlx_features" / "msas"
    if args.use_msa_server:
        if args.refresh_msa and cached_msa_dir.exists():
            shutil.rmtree(cached_msa_dir)
            print(f"[inference] --refresh-msa: cleared {cached_msa_dir}", flush=True)
        elif cached_msa_dir.is_dir() and any(cached_msa_dir.iterdir()):
            effective_use_msa_server = False
            effective_msa_directory = cached_msa_dir
            # Templates server can't reuse: chai-lab refuses to point
            # use_templates_server at an msa_directory (the server path
            # is the only thing that produces the ``all_chain_templates.m8``
            # file). Reuse the cached templates file instead if present.
            if args.use_templates_server:
                cached_templates = cached_msa_dir / "all_chain_templates.m8"
                if cached_templates.is_file():
                    effective_use_templates_server = False
                    effective_templates_path = cached_templates
                    print(
                        f"[inference] reusing cached MSAs + templates at "
                        f"{cached_msa_dir} (pass --refresh-msa to refetch)",
                        flush=True,
                    )
                else:
                    # Cached MSAs but no templates file. Safer to refetch
                    # everything than to silently drop templates.
                    shutil.rmtree(cached_msa_dir)
                    effective_use_msa_server = True
                    effective_msa_directory = None
                    print(
                        f"[inference] cached MSAs at {cached_msa_dir} have no "
                        "all_chain_templates.m8; refetching fresh since "
                        "--use-templates-server was requested",
                        flush=True,
                    )
            else:
                print(
                    f"[inference] reusing cached MSAs at {cached_msa_dir} "
                    "(pass --refresh-msa to refetch)",
                    flush=True,
                )

    print(
        f"[inference] featurizing {fasta_path} "
        f"(esm={args.esm_backend}, constraints={bool(args.constraint_path)}, "
        f"msa={'server' if effective_use_msa_server else ('dir' if effective_msa_directory else 'none')}, "
        f"templates={'server' if effective_use_templates_server else ('file' if effective_templates_path else 'none')}) ...",
        flush=True,
    )
    # Only pass msa_plot_path when we actually have an MSA source,
    # otherwise featurize_fasta's own msa_context.mask.any() guard skips
    # the plot silently anyway -- but wiring the path regardless is the
    # simplest thing: empty MSA + path set = no plot, the skip logic is
    # in featurize_fasta.
    msa_plot_path_arg = (
        (output_dir / "msa_coverage.png") if args.write_msa_plot else None
    )

    # Resolve the chai-1-style ``fasta_names_as_cif_chains`` default
    # here so downstream bookkeeping (CIF ref context, manifest) can
    # record the effective value, not just the user's None.
    if args.fasta_chain_names is None:
        effective_fasta_chain_names = args.constraint_path is not None
    else:
        effective_fasta_chain_names = bool(args.fasta_chain_names)

    ctx = featurize_fasta(
        fasta_path,
        output_dir=feature_dir / "mlx_features",
        constraint_path=args.constraint_path,
        msa_directory=effective_msa_directory,
        use_msa_server=effective_use_msa_server,
        msa_server_url=args.msa_server_url,
        use_templates_server=effective_use_templates_server,
        templates_path=effective_templates_path,
        esm_backend=args.esm_backend,
        esm_cache_dir=args.esm_cache_dir,
        msa_plot_path=msa_plot_path_arg,
        entity_name_as_subchain=effective_fasta_chain_names,
        pad_strategy=args.pad_strategy,
    )

    # Loop over trunk samples (chai-1 'num_trunk_samples' parity).
    # Each trunk sample uses seed = args.seed + trunk_idx and lands in
    # output_dir/trunk_<i>/ when num_trunk_samples > 1. A single trunk
    # sample stays flat under output_dir (matching chai-lab's own
    # "one trunk sample stays flat" layout).
    import numpy as np

    num_trunk_samples = max(1, int(args.num_trunk_samples))
    all_cif_paths: list[Path] = []
    all_per_sample_score_paths: list[str] = []
    all_aggregate_scores: list[float] = []
    per_trunk_manifests: list[dict] = []
    t_all = time.perf_counter()
    for trunk_idx in range(num_trunk_samples):
        trunk_seed = args.seed + trunk_idx
        trunk_output_dir = (
            output_dir / f"trunk_{trunk_idx}"
            if num_trunk_samples > 1
            else output_dir
        )
        trunk_output_dir.mkdir(parents=True, exist_ok=True)

        mx.random.seed(trunk_seed)
        print(
            f"[inference] running inference (trunk {trunk_idx + 1}/"
            f"{num_trunk_samples}, seed={trunk_seed}, recycles={args.recycles}, "
            f"steps={args.num_steps}, samples={args.num_samples}, "
            f"recycle_msa_subsample={args.recycle_msa_subsample}, "
            f"debug={args.debug}) ...",
            flush=True,
        )
        t_inf = time.perf_counter()
        if args.debug:
            result = model.run_inference_debug(
                ctx,
                recycles=args.recycles,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                recycle_msa_subsample=args.recycle_msa_subsample,
            )
        else:
            result = model.run_inference(
                ctx,
                recycles=args.recycles,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                recycle_msa_subsample=args.recycle_msa_subsample,
            )
        wall = time.perf_counter() - t_inf
        print(f"[inference] trunk {trunk_idx + 1} inference done in {wall:.1f}s",
              flush=True)

        coords_mx = result.coords.astype(mx.float32)
        coords_np = np.array(coords_mx)

        per_atom_plddt_np: "np.ndarray | None" = None
        if result.ranking.per_atom_plddt is not None:
            per_atom_plddt_np = np.array(
                result.ranking.per_atom_plddt.astype(mx.float32)
            )

        agg = np.array(result.ranking.aggregate_score.astype(mx.float32)).reshape(-1)
        best_idx = int(agg.argmax())
        best_score = float(agg[best_idx])
        print(
            f"[inference] coords shape={coords_np.shape}, best sample={best_idx} "
            f"(aggregate_score={best_score:.4f})",
            flush=True,
        )

        trunk_cif_paths: list[Path] = []
        if not args.skip_cif:
            print("[inference] writing CIFs ...", flush=True)
            t_cif = time.perf_counter()
            try:
                trunk_cif_paths = _save_cifs(
                    coords_np=coords_np,
                    per_atom_plddt_np=per_atom_plddt_np,
                    output_dir=trunk_output_dir,
                    fasta_path=fasta_path,
                    feature_dir=feature_dir,
                    entity_name_as_subchain=effective_fasta_chain_names,
                    pad_strategy=args.pad_strategy,
                )
            except Exception as exc:  # pragma: no cover - surfaced to the user
                print(
                    f"[inference] CIF export failed: {type(exc).__name__}: {exc}\n"
                    "[inference] falling back to --save-npz only; "
                    "re-run with --skip-cif to silence this error.",
                    file=sys.stderr,
                    flush=True,
                )
                trunk_cif_paths = []
            else:
                print(
                    f"[inference] wrote {len(trunk_cif_paths)} CIF(s) in "
                    f"{time.perf_counter() - t_cif:.1f}s",
                    flush=True,
                )

        scores_json = _scores_to_dict(result.ranking)
        scores_path = trunk_output_dir / "scores.json"
        scores_path.write_text(json.dumps(scores_json, indent=2))
        print(f"[inference] wrote scores -> {scores_path}", flush=True)

        # Per-sample npz sidecars matching chai-lab's upstream output
        # layout (scores.model_idx_{i}.npz, with per-token pae/pde/
        # plddt tensors from StructureCandidates).
        trunk_per_sample_score_paths: list[str] = _write_per_sample_scores(
            output_dir=trunk_output_dir,
            ranking=result.ranking,
            confidence=result.confidence,
            structure=ctx.structure_inputs,
            num_samples=coords_np.shape[1],
        )

        trunk_manifest = {
            "trunk_index": trunk_idx,
            "seed": trunk_seed,
            "output_dir": str(trunk_output_dir),
            "best_sample_index": best_idx,
            "best_aggregate_score": best_score,
            "wall_seconds": wall,
            "cif_paths": [str(p) for p in trunk_cif_paths],
            "per_sample_score_paths": trunk_per_sample_score_paths,
        }
        per_trunk_manifests.append(trunk_manifest)
        all_cif_paths.extend(trunk_cif_paths)
        all_per_sample_score_paths.extend(trunk_per_sample_score_paths)
        all_aggregate_scores.extend(agg.tolist())

        # --save-npz is a single-trunk-single-FASTA convenience flag.
        # When multi-trunk or batch mode, the per-sample npz sidecars
        # are the canonical tensor dump.
        if (
            args.save_npz is not None
            and args.fasta_dir is None
            and num_trunk_samples == 1
        ):
            args.save_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.save_npz,
                coords=coords_np,
                aggregate_score=np.array(
                    result.ranking.aggregate_score.astype(mx.float32)
                ),
                ptm=np.array(result.ranking.ptm.astype(mx.float32)),
                iptm=np.array(result.ranking.iptm.astype(mx.float32)),
                best_index=best_idx,
                best_score=best_score,
            )
            print(f"[inference] wrote raw tensors -> {args.save_npz}",
                  flush=True)

        # Free the per-trunk tensors before the next trunk sample so
        # memory doesn't accumulate across trunk samples.
        del result, coords_mx, coords_np
        mx.clear_cache()

    wall_total = time.perf_counter() - t_all

    # Top-level best across all trunk samples (chai-1
    # ``StructureCandidates.sorted()`` picks the highest aggregate).
    agg_arr = np.asarray(all_aggregate_scores, dtype=np.float32)
    if agg_arr.size > 0:
        global_best_idx = int(agg_arr.argmax())
        global_best_score = float(agg_arr[global_best_idx])
    else:
        global_best_idx = -1
        global_best_score = float("nan")

    manifest = {
        "fasta": str(fasta_path),
        "output_dir": str(output_dir),
        "weights_dir": str(args.weights_dir),
        "dtype": args.dtype,
        "seed": args.seed,
        "num_recycles": args.recycles,
        "num_steps": args.num_steps,
        "num_samples": args.num_samples,
        "num_trunk_samples": num_trunk_samples,
        "recycle_msa_subsample": int(args.recycle_msa_subsample),
        "pad_strategy": args.pad_strategy,
        "esm_backend": args.esm_backend,
        "esm_cache_dir": str(args.esm_cache_dir) if args.esm_cache_dir else None,
        "constraint_path": str(args.constraint_path) if args.constraint_path else None,
        "msa_directory_requested": str(args.msa_directory) if args.msa_directory else None,
        "msa_directory_effective": str(effective_msa_directory) if effective_msa_directory else None,
        "use_msa_server_requested": bool(args.use_msa_server),
        "use_msa_server_effective": bool(effective_use_msa_server),
        "msa_server_url": args.msa_server_url if effective_use_msa_server else None,
        "refresh_msa": bool(args.refresh_msa),
        "templates_path_requested": str(args.templates_path) if args.templates_path else None,
        "templates_path_effective": str(effective_templates_path) if effective_templates_path else None,
        "use_templates_server_requested": bool(args.use_templates_server),
        "use_templates_server_effective": bool(effective_use_templates_server),
        "fasta_chain_names_requested": args.fasta_chain_names,
        "fasta_chain_names_effective": bool(effective_fasta_chain_names),
        # When num_trunk_samples > 1, best_sample_index is the index
        # across the concatenated candidate list in ``cif_paths``.
        "best_sample_index": global_best_idx,
        "best_aggregate_score": global_best_score,
        "wall_seconds": wall_total,
        "weights_load_seconds": weights_load_seconds,
        "cif_paths": [str(p) for p in all_cif_paths],
        "per_sample_score_paths": all_per_sample_score_paths,
        "trunks": per_trunk_manifests if num_trunk_samples > 1 else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[inference] wrote manifest -> {manifest_path}", flush=True)

    print(f"[inference] done -> {output_dir}", flush=True)
    return manifest


def main(argv: "list[str] | None" = None) -> None:
    args = _parse_args(argv)

    mx.random.seed(args.seed)

    from chai_mlx.data.fasta import validate_fasta_or_raise

    # Collect the list of FASTAs we're going to fold. In single-FASTA
    # mode this is just [args.fasta]; in batch mode we enumerate the
    # directory and validate every file up-front so the user doesn't
    # wait for model load just to hit a bad header 4 files in.
    if args.fasta is not None:
        fasta_list = [args.fasta]
    else:
        fasta_list = sorted(args.fasta_dir.glob("*.fasta"))

    # Fail fast on obvious FASTA problems before paying the model-load
    # cost. In batch mode we run validation against every file and
    # collect issues, so the user can fix them all at once.
    validation_errors: list[tuple[Path, str]] = []
    for fp in fasta_list:
        try:
            validate_fasta_or_raise(fp)
        except SystemExit as exc:  # raised with a multi-line message
            validation_errors.append((fp, str(exc)))

    if validation_errors and len(fasta_list) == 1:
        # Single-file mode: preserve the original behaviour of printing
        # the exact error and exiting non-zero.
        raise SystemExit(validation_errors[0][1])
    if validation_errors:
        # Batch mode: print each failure but continue with the valid
        # ones. Fully failing the batch would be surprising.
        for fp, msg in validation_errors:
            print(
                f"[inference] skipping {fp}: {msg}",
                file=sys.stderr, flush=True,
            )
        valid = [fp for fp in fasta_list if fp not in {e[0] for e in validation_errors}]
        if not valid:
            raise SystemExit("[inference] no valid FASTAs to process, aborting.")
        fasta_list = valid

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[inference] loading model from {args.weights_dir} (dtype={args.dtype}) ...",
          flush=True)
    t_load = time.perf_counter()
    model = ChaiMLX.from_pretrained(
        args.weights_dir, strict=False, compute_dtype=args.dtype,
    )
    t_load = time.perf_counter() - t_load
    print(f"[inference] model loaded in {t_load:.1f}s", flush=True)

    if args.fasta is not None:
        # Single-FASTA mode: the user's --output-dir IS the run dir.
        feature_dir = args.feature_dir or (args.output_dir / "_features")
        feature_dir.mkdir(parents=True, exist_ok=True)
        _fold_one_fasta(
            model=model,
            args=args,
            fasta_path=args.fasta,
            output_dir=args.output_dir,
            feature_dir=feature_dir,
            weights_load_seconds=t_load,
        )
        return

    # Batch mode: --output-dir holds one subdir per FASTA, plus a
    # top-level run_summary.json aggregating per-file status.
    print(f"[inference] batch mode: {len(fasta_list)} FASTA(s) under {args.fasta_dir}",
          flush=True)
    run_summary: list[dict] = []
    for i, fasta_path in enumerate(fasta_list, start=1):
        stem = fasta_path.stem
        sub_output = args.output_dir / stem
        sub_output.mkdir(parents=True, exist_ok=True)
        sub_feature = (args.feature_dir / stem) if args.feature_dir else (sub_output / "_features")
        sub_feature.mkdir(parents=True, exist_ok=True)
        print(f"[inference] [{i}/{len(fasta_list)}] -> {stem}", flush=True)
        t0 = time.perf_counter()
        try:
            manifest = _fold_one_fasta(
                model=model,
                args=args,
                fasta_path=fasta_path,
                output_dir=sub_output,
                feature_dir=sub_feature,
                weights_load_seconds=t_load,
            )
            status = "ok"
            err = None
        except Exception as exc:
            status = "fail"
            err = f"{type(exc).__name__}: {exc}"
            manifest = None
            print(f"[inference]   FAILED: {err}", file=sys.stderr, flush=True)
        run_summary.append({
            "fasta": str(fasta_path),
            "output_dir": str(sub_output),
            "status": status,
            "error": err,
            "wall_seconds": time.perf_counter() - t0,
            "best_aggregate_score": (
                manifest["best_aggregate_score"] if manifest else None
            ),
        })

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps({
        "weights_dir": str(args.weights_dir),
        "dtype": args.dtype,
        "seed": args.seed,
        "num_recycles": args.recycles,
        "num_steps": args.num_steps,
        "num_samples": args.num_samples,
        "esm_backend": args.esm_backend,
        "weights_load_seconds": t_load,
        "runs": run_summary,
    }, indent=2))
    ok = sum(1 for r in run_summary if r["status"] == "ok")
    print(f"[inference] batch done: {ok}/{len(run_summary)} ok -> {summary_path}",
          flush=True)


if __name__ == "__main__":
    main()
