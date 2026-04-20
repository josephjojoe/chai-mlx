"""Frontend featurization adapters.

``featurize()`` accepts precomputed tensors (the fast path for callers who
already have encoded features).

``featurize_fasta()`` delegates to **chai-lab's** featurization pipeline for
correctness, then converts the batch dict into a ``FeatureContext`` that the
MLX model consumes.  This avoids reimplementing the 30+ feature generators
and their encoding quirks.

When ``featurize_fasta()`` is used, raw per-feature tensors are stored as
``FeatureContext.raw_features`` so that ``FeatureEmbedding`` can encode,
concatenate, and project each feature group independently — avoiding the
multi-GB materialisation of a single wide encoded tensor that the
pre-computed path requires.

The ``esm_backend`` knob on :func:`featurize_fasta` selects how ESM-2
embeddings are obtained:

* ``"off"`` (default): no ESM embeddings; the feature is zero-filled.
  Matches every existing harness call site.
* ``"chai"``: pass ``use_esm_embeddings=True`` through to chai-lab, which
  loads its traced CUDA fp16 checkpoint.  Only sensible on a machine that
  has torch + CUDA.
* ``"mlx"``: pre-compute embeddings with ``esm-mlx`` on Apple silicon
  and inject them into the chai-lab featurization pipeline.  Requires
  the ``[esm]`` extra.

The ``pad_strategy`` knob on :func:`featurize_fasta` (and on the
matching helpers in :mod:`chai_mlx.cli`) selects how the token / atom
axes are padded:

* ``"exact"`` (default): pad to the smallest shape the MLX kernels
  accept — ``n_tokens = num_tokens`` exactly, and ``n_atoms`` rounded
  up to the next multiple of 32 (the query-block stride of the local
  atom attention; see ``chai_lab.model.utils.get_qkv_indices_for_blocks``).
  This skips the multi-hundred-token padding that chai-lab's seven
  static buckets impose on shorter inputs.
* ``"bucket"``: legacy chai-lab behaviour — pad ``n_tokens`` up to the
  smallest value in ``AVAILABLE_MODEL_SIZES = [256, 384, 512, 768, 1024,
  1536, 2048]`` that fits, and set ``n_atoms = 23 * n_tokens``. Necessary
  for parity comparisons with the CUDA reference bundle's traced
  TorchScript artefacts (which were exported at those seven sizes; see
  ``drift_attribution.md``).

The MLX model forward itself does not depend on either choice — it reads
``num_tokens`` and ``num_atoms`` from the input tensor shapes at call
time. The only hard constraint is ``n_atoms % 32 == 0``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from .types import FeatureContext, InputBundle, StructureInputs

# Query-block stride of the local atom attention. Any padded atom count
# must be a multiple of this -- see ``chai_lab.model.utils``
# ``get_qkv_indices_for_blocks``'s ``sequence_length % stride == 0``
# assertion, and ``chai_mlx/nn/layers/atom_attention.py``'s
# ``num_blocks = a // 32`` reshape.
_ATOM_BLOCK_STRIDE: int = 32

PadStrategy = Literal["exact", "bucket"]

_REQUIRED_KEYS = {
    "token_features",
    "token_pair_features",
    "atom_features",
    "atom_pair_features",
    "msa_features",
    "template_features",
    "structure_inputs",
}


# Input-space ceilings enforced by chai-1 upstream (see
# ``chai_lab.data.dataset.all_atom_feature_context.MAX_MSA_DEPTH`` and
# ``MAX_NUM_TEMPLATES``, plus ``chai_lab.data.collate.utils.AVAILABLE_MODEL_SIZES``).
# Mirrored here so callers get a loud ``UnsupportedInputError`` at
# featurize time instead of a cryptic OOM or shape-mismatch inside
# the trunk. We pull the live values from chai-lab when it's
# importable and fall back to the literal constants if not (the
# fallback is only exercised when someone calls ``featurize()`` on
# precomputed tensors without chai-lab installed; the numbers are the
# same constants chai-lab ships).
_MAX_MSA_DEPTH_FALLBACK: int = 16_384
_MAX_NUM_TEMPLATES_FALLBACK: int = 4
_SUPPORTED_CROP_SIZES_FALLBACK: tuple[int, ...] = (
    256, 384, 512, 768, 1024, 1536, 2048,
)


class UnsupportedInputError(ValueError):
    """Raised when a FASTA exceeds chai-1's hard input-space limits.

    Matches the name chai-1 upstream uses so callers catching it
    don't need to distinguish ``chai_lab`` vs ``chai_mlx`` sources.
    """


def _chai_lab_limits() -> tuple[int, int, tuple[int, ...]]:
    """Return ``(MAX_MSA_DEPTH, MAX_NUM_TEMPLATES, AVAILABLE_MODEL_SIZES)``.

    Prefers the live values from chai-lab; falls back to the constants
    that shipped at the pinned commit if chai-lab is unimportable.
    """
    try:
        from chai_lab.data.dataset.all_atom_feature_context import (
            MAX_MSA_DEPTH, MAX_NUM_TEMPLATES,
        )
        from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
        return int(MAX_MSA_DEPTH), int(MAX_NUM_TEMPLATES), tuple(AVAILABLE_MODEL_SIZES)
    except ImportError:
        return (
            _MAX_MSA_DEPTH_FALLBACK,
            _MAX_NUM_TEMPLATES_FALLBACK,
            _SUPPORTED_CROP_SIZES_FALLBACK,
        )


def _enforce_input_limits(feature_context) -> None:
    """Raise :class:`UnsupportedInputError` if *feature_context* exceeds
    chai-1's hard input-space limits.

    Called inside :func:`featurize_fasta` immediately after chai-lab
    produces its ``AllAtomFeatureContext`` and before the MLX-side
    collate step. The three checks mirror
    :func:`chai_lab.chai1.raise_if_too_many_tokens`,
    :func:`raise_if_too_many_templates`, and
    :func:`raise_if_msa_too_deep` so users see the same "fail fast at
    featurize" behaviour regardless of which side is running.

    The token-count check only enforces the architectural upper bound
    (``max(AVAILABLE_MODEL_SIZES) = 2048``) -- chai-lab's intermediate
    crop sizes are padding targets for its traced TorchScript artefacts
    and are listed only as a hint when the error fires.
    """
    max_msa, max_templates, crop_sizes = _chai_lab_limits()

    n_tokens = feature_context.structure_context.num_tokens
    largest_crop = max(crop_sizes)
    if n_tokens > largest_crop:
        raise UnsupportedInputError(
            f"Too many tokens in input: {n_tokens} > {largest_crop}. "
            "Please limit the length of the input sequence. "
            "(The MLX port runs at the exact input length; this ceiling "
            "comes from chai-1's reference bundle, which was traced at "
            f"crop sizes {sorted(crop_sizes)} with 2048 as the largest.)"
        )

    n_templates = feature_context.template_context.num_templates
    if n_templates > max_templates:
        raise UnsupportedInputError(
            f"Too many templates in input: {n_templates} > {max_templates}. "
            "Please limit the number of templates."
        )

    msa_depth = feature_context.msa_context.depth
    if msa_depth > max_msa:
        raise UnsupportedInputError(
            f"MSA too deep: {msa_depth} > {max_msa}. "
            "Please limit the MSA depth (consider setting "
            "recycle_msa_subsample to subsample at recycle time)."
        )


# ---------------------------------------------------------------------------
# Pad-size override for exact-length inference
# ---------------------------------------------------------------------------


def _exact_pad_size(n_tokens: int, n_atoms: int) -> tuple[int, int]:
    """Return the tightest ``(n_tokens, n_atoms)`` the MLX stack accepts.

    * Token axis: no divisibility constraint in the MLX forward path --
      pair-feature chunks loop dynamically and atom-token segment_mean
      reads the count at call time -- so we pick ``n_tokens`` exactly.
    * Atom axis: ``get_qkv_indices_for_blocks`` asserts
      ``sequence_length % 32 == 0`` and the local atom-attention blocks
      reshape ``a`` into ``a // 32`` groups of 32, so pad up to the next
      multiple of 32.
    """
    stride = _ATOM_BLOCK_STRIDE
    padded_atoms = ((n_atoms + stride - 1) // stride) * stride
    return n_tokens, max(padded_atoms, stride)


@contextmanager
def _override_pad_strategy(strategy: PadStrategy) -> Iterator[None]:
    """Context manager that patches chai-lab's pad selector for exact-length.

    When ``strategy == "exact"`` we replace
    ``chai_lab.data.collate.utils.get_pad_sizes`` with a shim that reads
    the real structure sizes off each ``AllAtomFeatureContext`` and
    returns the tightest shape the MLX stack accepts (``n_tokens``
    verbatim, ``n_atoms`` rounded up to the next multiple of 32).

    When ``strategy == "bucket"`` this is a no-op and the original
    chai-lab selector runs, matching the reference bundle's seven
    TorchScript sizes and the ``23 * n_tokens`` atom bound.

    Both ``Collate._collate`` (in :func:`featurize_fasta`) and the
    CIF-writer ref context in :func:`chai_mlx.cli.infer._save_cifs` /
    :mod:`chai_mlx.cli.sweep_impl` must use the same strategy so the
    coords array emitted by the MLX model lines up with the atom
    bookkeeping that ``save_to_cif`` walks.
    """
    if strategy not in ("exact", "bucket"):
        raise ValueError(
            f"pad_strategy must be 'exact' or 'bucket'; got {strategy!r}"
        )
    if strategy == "bucket":
        yield
        return

    from chai_lab.data.collate import utils as collate_utils

    # CRITICAL: import ``collate`` BEFORE we patch ``utils.get_pad_sizes``.
    # ``chai_lab.data.collate.collate`` does ``from chai_lab.data.collate.utils
    # import get_pad_sizes`` at module-load time, so whatever value
    # ``collate_utils.get_pad_sizes`` has when ``collate.py`` is first
    # imported is what gets bound into ``collate_mod.get_pad_sizes`` --
    # and that binding is what ``Collate._collate`` actually resolves
    # at call time (via its ``__globals__``).  If we patched ``utils``
    # first and then triggered a lazy import of ``collate``, the
    # ``original_collate_ref`` we captured would be our patched shim,
    # and the ``finally`` restore would leave ``collate_mod`` stuck on
    # the patched function forever -- silently breaking every
    # subsequent bucket-mode call in the same process.
    try:
        from chai_lab.data.collate import collate as collate_mod
        collate_mod_had_ref = hasattr(collate_mod, "get_pad_sizes")
        original_collate_ref = (
            collate_mod.get_pad_sizes if collate_mod_had_ref else None
        )
    except ImportError:
        collate_mod = None  # type: ignore[assignment]
        collate_mod_had_ref = False
        original_collate_ref = None

    original = collate_utils.get_pad_sizes

    def _exact_get_pad_sizes(contexts):
        max_n_tokens = max(c.num_tokens for c in contexts)
        max_n_atoms = max(c.num_atoms for c in contexts)
        n_tokens, n_atoms = _exact_pad_size(max_n_tokens, max_n_atoms)
        return collate_utils.PadSizes(n_tokens=n_tokens, n_atoms=n_atoms)

    collate_utils.get_pad_sizes = _exact_get_pad_sizes
    if collate_mod_had_ref:
        collate_mod.get_pad_sizes = _exact_get_pad_sizes

    try:
        yield
    finally:
        collate_utils.get_pad_sizes = original
        if collate_mod_had_ref:
            collate_mod.get_pad_sizes = original_collate_ref


# ---------------------------------------------------------------------------
# Optional-dependency error surface helpers
# ---------------------------------------------------------------------------

def _require_chai_lab():
    """Import ``chai_lab.chai1`` or raise a readable RuntimeError.

    The bare ``ImportError`` from the inline ``from chai_lab.chai1
    import ...`` is unhelpful for users who forgot the ``[featurize]``
    extra. This helper re-raises with installation instructions.
    """
    try:
        import chai_lab.chai1 as chai1  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "featurize_fasta requires chai_lab. Install with:\n"
            "    pip install 'chai-mlx[featurize]'\n"
            "(or run from a git clone with the submodule checked out)."
        ) from exc
    return chai1


def _require_torch():
    """Import ``torch`` or raise a readable RuntimeError.

    ``torch`` is pulled in through the same ``[featurize]`` extra as
    chai_lab; surfacing the same instruction keeps the failure mode
    uniform.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "featurize_fasta requires torch. Install with:\n"
            "    pip install 'chai-mlx[featurize]'"
        ) from exc
    return torch


# 20 GB captures the "16 GB unified memory Mac" OOM case from HANDOFF
# §2.2 while leaving comfortable headroom on machines just above. ESM-2
# 3B resident at fp32 is ~11 GB; add chai-mlx's ~1.2 GB weights and
# several GB of trunk activations and we're well past 16 GB. Users who
# know better can ignore the warning -- this is print-only, never raise.
_ESM_MLX_MIN_RAM_BYTES: int = 20 * 1024 ** 3


def _warn_if_insufficient_ram_for_esm_mlx() -> None:
    """Print a stderr warning if this host probably can't hold ESM-3B.

    ``psutil`` is a soft dep -- installed on most dev machines but not
    in the hard-required list. Missing ``psutil`` silently skips the
    check; the user will still see a Metal allocator abort if they OOM.
    """
    import sys
    try:
        import psutil
    except ImportError:
        return
    try:
        total = int(psutil.virtual_memory().total)
    except Exception:  # pragma: no cover - defensive
        return
    if total < _ESM_MLX_MIN_RAM_BYTES:
        gib = total / (1024 ** 3)
        print(
            f"[chai-mlx] warning: esm_backend='mlx' loads ~11 GB of ESM-2 3B "
            f"weights in-process; this machine has {gib:.1f} GiB total RAM. "
            "Prefer esm_backend='mlx_cache' after pre-computing embeddings "
            "with scripts/precompute_esm_mlx.py (see HANDOFF.md §2.2).",
            file=sys.stderr,
            flush=True,
        )


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

    Accepts precomputed ``FeatureContext``, ``InputBundle``, or raw dict of
    tensors.  For FASTA-based featurization, use ``featurize_fasta()`` instead.
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


# ---------------------------------------------------------------------------
# FASTA-based featurization via chai-lab
# ---------------------------------------------------------------------------


def reuse_msa_dir_if_present(output_dir: str | Path) -> Path | None:
    """Return the chai-lab MSA cache directory under *output_dir* if it exists.

    ``chai_lab.chai1.make_all_atom_feature_context`` creates
    ``<output_dir>/msas`` with ``exist_ok=False`` whenever
    ``use_msa_server=True``, so the second call for the same output
    directory crashes. Callers can use this helper to detect a prior
    cache and pass it as ``msa_directory=`` instead of setting
    ``use_msa_server=True`` again::

        cached = reuse_msa_dir_if_present(output_dir)
        ctx = featurize_fasta(
            fasta,
            output_dir=output_dir,
            msa_directory=cached,             # offline reuse when available
            use_msa_server=(cached is None),  # otherwise online fetch
        )

    Returns ``None`` if no cache exists or the directory is empty.
    """
    msa_dir = Path(output_dir) / "msas"
    if msa_dir.is_dir() and any(msa_dir.iterdir()):
        return msa_dir
    return None


def featurize_fasta(
    fasta_file: str | Path,
    *,
    output_dir: str | Path | None = None,
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_esm_embeddings: bool | None = None,
    esm_backend: Literal["off", "chai", "mlx", "mlx_cache"] = "off",
    esm_cache_dir: str | Path | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    use_templates_server: bool = False,
    templates_path: Path | None = None,
    msa_plot_path: Path | None = None,
    entity_name_as_subchain: bool | None = None,
    pad_strategy: PadStrategy = "exact",
) -> FeatureContext:
    """Full FASTA-to-FeatureContext entry point using chai-lab's pipeline.

    Requires ``chai-lab`` and ``torch`` to be installed.  Returns a
    ``FeatureContext`` with ``raw_features`` populated — the heavy-duty
    encoding + projection is deferred to ``FeatureEmbedding.__call__``.

    Parameters
    ----------
    esm_backend:
        * ``"off"`` (default) — zero-fill ESM embeddings. All existing
          harness scripts use this.
        * ``"chai"`` — delegate to chai-lab's traced CUDA fp16 checkpoint.
          Only sensible on a CUDA host.
        * ``"mlx"`` — compute embeddings with ``esm-mlx`` in-process.
          Loads the full 3B model; not advisable alongside chai-mlx
          inference on a 16 GB Mac.
        * ``"mlx_cache"`` — load embeddings from the directory pointed to
          by ``esm_cache_dir`` (as produced by
          :mod:`scripts.precompute_esm_mlx`). Zero extra RAM cost at
          inference time.

    esm_cache_dir:
        Directory of pre-computed ``<sha1>.npy`` files. Required when
        ``esm_backend="mlx_cache"``; ignored otherwise.

    use_esm_embeddings:
        **Deprecated.** Retained for backward compatibility with callers
        that pass this boolean.  ``True`` is equivalent to
        ``esm_backend="chai"``; ``False`` is equivalent to the default
        ``esm_backend="off"``.  Passing both raises ``ValueError``.

    pad_strategy:
        * ``"exact"`` (default) — pad the token axis to exactly
          ``num_tokens`` and the atom axis to the smallest multiple of
          32 that fits. Recommended for all MLX-only workflows: skips
          the 100–400 tokens of dead padding that chai-lab's seven
          static buckets impose on typical inputs.
        * ``"bucket"`` — pad up to the smallest value in
          ``chai_lab.data.collate.utils.AVAILABLE_MODEL_SIZES``
          (``[256, 384, 512, 768, 1024, 1536, 2048]``), with
          ``n_atoms = 23 * n_tokens``. Necessary for parity comparisons
          with the CUDA reference bundle's TorchScript artefacts, which
          were traced at those exact seven sizes. Downstream helpers
          (``chai_mlx.cli.infer._save_cifs``,
          :mod:`chai_mlx.cli.sweep_impl`) must be told the same value
          so the CIF atom-indexing matches the MLX coords tensor.
    """
    import tempfile

    # Patch chai-lab's RDKit timeout on macOS before its modules are
    # imported inside ``make_all_atom_feature_context``.
    from chai_mlx.data._rdkit_timeout_patch import apply_rdkit_timeout_patch

    apply_rdkit_timeout_patch()

    fasta_file = Path(fasta_file)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="chai_mlx_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if use_esm_embeddings is not None:
        if esm_backend != "off":
            raise ValueError(
                "Pass only one of use_esm_embeddings= or esm_backend= "
                "(use_esm_embeddings is deprecated)"
            )
        esm_backend = "chai" if use_esm_embeddings else "off"

    if esm_backend not in ("off", "chai", "mlx", "mlx_cache"):
        raise ValueError(
            f"esm_backend must be 'off', 'chai', 'mlx', or 'mlx_cache'; got {esm_backend!r}"
        )
    if esm_backend == "mlx_cache" and esm_cache_dir is None:
        raise ValueError("esm_backend='mlx_cache' requires esm_cache_dir=")

    if esm_backend == "mlx":
        _warn_if_insufficient_ram_for_esm_mlx()

    # ``entity_name_as_subchain`` toggles whether chai-lab uses FASTA
    # entity names as chain IDs.  Chai-1 upstream defaults to False
    # (sequential A/B/C... labels in the output CIF); constraint CSVs
    # reference chains *by FASTA name*, so when the user attaches a
    # CSV we force it to True so the restraint lookups line up with
    # what the user wrote in the FASTA.  Outside the CSV case we
    # match chai-1's default for drop-in compatibility with downstream
    # tooling that expects A/B/C... chain IDs.
    if entity_name_as_subchain is None:
        entity_name_as_subchain = constraint_path is not None

    chai1 = _require_chai_lab()
    Collate = chai1.Collate
    TokenBondRestraint = chai1.TokenBondRestraint
    feature_factory = chai1.feature_factory
    make_all_atom_feature_context = chai1.make_all_atom_feature_context

    feature_context = make_all_atom_feature_context(
        fasta_file,
        output_dir=output_dir,
        entity_name_as_subchain=entity_name_as_subchain,
        use_esm_embeddings=(esm_backend == "chai"),
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        templates_path=templates_path,
    )

    # Enforce chai-1's hard input-space limits up-front so users see a
    # loud UnsupportedInputError here rather than a cryptic OOM or
    # shape-mismatch inside the trunk.
    _enforce_input_limits(feature_context)

    if esm_backend == "mlx":
        from chai_mlx.data.esm_mlx_adapter import build_embedding_context

        feature_context.embedding_context = build_embedding_context(
            feature_context.chains
        )
    elif esm_backend == "mlx_cache":
        from chai_mlx.data.esm_mlx_adapter import build_embedding_context_from_cache

        feature_context.embedding_context = build_embedding_context_from_cache(
            feature_context.chains, cache_dir=esm_cache_dir
        )

    # Optional MSA coverage plot (matches chai-lab's ``chai1.run_inference``
    # behaviour). Only drawn when the MSA is non-empty; skipped silently
    # when matplotlib is not installed so the plot flag is safe to pass
    # even on slim envs.
    if msa_plot_path is not None and feature_context.msa_context.mask.any():
        try:
            from chai_lab.utils.plot import plot_msa
            msa_plot_path = Path(msa_plot_path)
            msa_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_msa(
                input_tokens=feature_context.structure_context.token_residue_type,
                msa_tokens=feature_context.msa_context.tokens,
                out_fname=msa_plot_path,
            )
        except ImportError:
            # matplotlib not installed; silently skip.
            pass

    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )
    with _override_pad_strategy(pad_strategy):
        batch = collator([feature_context])

        # ``TokenBondRestraint.generate`` doesn't itself call
        # ``get_pad_sizes``, but it reads ``batch["inputs"][...]`` whose
        # shapes depend on the collator above. Keep it inside the
        # override scope for safety.
        bond_ft = TokenBondRestraint().generate(batch=batch).data

    return _batch_to_feature_context(batch, bond_ft)


# ---------------------------------------------------------------------------
# Batch-dict → FeatureContext conversion
# ---------------------------------------------------------------------------

def _batch_to_feature_context(
    batch: dict[str, Any],
    bond_features: Any,
) -> FeatureContext:
    """Convert a chai-lab batch dict into a ``FeatureContext``.

    Rather than encoding and concatenating into wide dense tensors on the
    CPU, we simply convert each raw feature tensor to MLX and store them
    in ``raw_features``.  The ``FeatureEmbedding`` will encode + project
    one group at a time, limiting peak memory to the largest single group
    rather than the sum of all groups.

    The wide ``*_features`` fields are set to zero-sized placeholders since
    they are unused when ``raw_features`` is present.
    """
    torch = _require_torch()
    chai1 = _require_chai_lab()
    feature_generators = chai1.feature_generators

    import mlx.core as mx

    features = batch["features"]
    inputs = batch["inputs"]

    def _mx(t: torch.Tensor) -> mx.array:
        return mx.array(t.detach().cpu().numpy())

    raw: dict[str, mx.array] = {}
    for name in feature_generators:
        raw[name] = _mx(features[name])

    # -- build StructureInputs ------------------------------------------

    token_exists = inputs["token_exists_mask"]
    atom_exists = inputs["atom_exists_mask"]
    token_pair_mask = torch.einsum("bi,bj->bij", token_exists.float(), token_exists.float())

    from chai_lab.data.parsing.structure.entity_type import EntityType as ChaiEntityType

    polymer_values = {
        ChaiEntityType.PROTEIN.value,
        ChaiEntityType.RNA.value,
        ChaiEntityType.DNA.value,
    }
    entity_type = inputs["token_entity_type"].long()
    is_polymer = torch.zeros_like(entity_type, dtype=torch.float32)
    for v in polymer_values:
        is_polymer[entity_type == v] = 1.0

    B = token_exists.shape[0]

    structure = StructureInputs(
        atom_exists_mask=_mx(atom_exists.float()),
        token_exists_mask=_mx(token_exists.float()),
        token_pair_mask=_mx(token_pair_mask),
        atom_token_index=_mx(inputs["atom_token_index"].long()),
        atom_within_token_index=_mx(inputs["atom_within_token_index"].long()),
        token_reference_atom_index=_mx(inputs["token_ref_atom_index"].long()),
        token_centre_atom_index=_mx(inputs["token_centre_atom_index"].long()),
        token_asym_id=_mx(inputs["token_asym_id"].long()),
        token_entity_id=_mx(inputs["token_entity_id"].long()),
        token_chain_id=_mx(inputs["token_asym_id"].long()),
        token_is_polymer=_mx(is_polymer),
        atom_ref_positions=_mx(inputs["atom_ref_pos"].float()),
        atom_ref_space_uid=_mx(inputs["atom_ref_space_uid"].long()),
        atom_q_indices=_mx(inputs["block_atom_pair_q_idces"].unsqueeze(0).expand(B, -1, -1)
                           if inputs["block_atom_pair_q_idces"].dim() == 2
                           else inputs["block_atom_pair_q_idces"]),
        atom_kv_indices=_mx(inputs["block_atom_pair_kv_idces"].unsqueeze(0).expand(B, -1, -1)
                            if inputs["block_atom_pair_kv_idces"].dim() == 2
                            else inputs["block_atom_pair_kv_idces"]),
        block_atom_pair_mask=_mx(inputs["block_atom_pair_mask"].float()),
        msa_mask=_mx(inputs["msa_mask"]),
        template_input_masks=_mx(
            torch.einsum(
                "btn,btm->btnm",
                inputs["template_mask"].float(),
                inputs["template_mask"].float(),
            )
        ),
        token_residue_index=_mx(inputs["token_residue_index"].long()),
        token_entity_type=_mx(inputs["token_entity_type"].long()),
        token_backbone_frame_mask=_mx(inputs["token_backbone_frame_mask"]),
        token_backbone_frame_index=_mx(inputs["token_backbone_frame_index"].long()),
    )

    N = token_exists.shape[1]
    empty = mx.zeros((B, 0))

    return FeatureContext(
        token_features=empty,
        token_pair_features=empty,
        atom_features=empty,
        atom_pair_features=empty,
        msa_features=empty,
        template_features=empty,
        structure_inputs=structure,
        bond_adjacency=_mx(bond_features),
        raw_features=raw,
    )
