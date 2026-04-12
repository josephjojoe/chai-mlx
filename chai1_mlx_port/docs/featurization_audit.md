# Featurization Re-Usage Tooling Audit

Rigorous evaluation of `featurize.py`, `embeddings.py`, `types.py`, and the
surrounding weight-loading infrastructure against the original chai-lab code
and the TorchScript `feature_embedding_forward256.py` ground truth.

---

## Architecture Overview

The port's strategy is sound in principle: delegate all data preparation (FASTA
parsing, tokenization, MSA/template/ESM loading, 30+ feature generators,
collation) to the upstream chai-lab package, then convert the resulting batch
dict into a `FeatureContext` of pre-encoded dense tensors that a simple
`nn.Linear`-based `FeatureEmbedding` can project.

The pipeline is:

```
chai-lab featurization  →  _batch_to_feature_context()  →  FeatureContext
FeatureContext           →  FeatureEmbedding (Linear projections)
                         →  BondProjection
                         →  TokenInputEmbedding (atom encoder + outer sum)
```

This mirrors the original TorchScript split:

```
feature generators  →  feature_embedding.pt  →  embedded features dict
                        (per-feature encoding + Linear projection)
```

The critical difference: in the TorchScript, encoding (one-hot expansion, RBF,
OUTERSUM embedding) happens *inside* `feature_embedding.pt`, not before it.
The port must replicate that encoding in `_batch_to_feature_context()` so its
simpler `FeatureEmbedding` (just Linears) receives the same input.

---

## Issue 1 — CRITICAL: ONE_HOT mask channel omission

**Files:** `featurize.py:_encode_one_hot`, verified against
`feature_embedding_forward256.py`.

For every ONE_HOT feature with `can_mask=True`, the TorchScript uses
`one_hot(idx, num_classes + 1)`, allocating an extra channel for the mask
sentinel (index = `num_classes`).  The port's `_encode_one_hot` uses
`one_hot(idx, num_classes)` and `clamp(0, num_classes - 1)`, which:

1. **Clips masked indices** to the highest valid class, losing mask information.
2. **Produces fewer channels**, causing every subsequent feature in the
   concatenation to be shifted, misaligning all Linear weight columns.

Affected features and the per-feature dimension error:

| Feature | num_classes | TorchScript OH | Port OH | Δ |
|---------|------------|----------------|---------|---|
| ResidueType | 32 | 33 | 32 | −1 |
| IsDistillation | 1 | 2 | 1 | −1 |
| TokenBFactor | 2 | 3 | 2 | −1 |
| TokenPLDDT | 3 | 4 | 3 | −1 |
| AtomNameOneHot | 64 | 65 (×4 chars) | 64 (×4) | −4 |
| AtomRefElement | 129 | 130 | 129 | −1 |
| BlockedAtomPairDistogram | 11 | 12 | 11 | −1 |
| DockingConstraintGenerator | 5 | 6 | 5 | −1 |
| TemplateDistogram | 38 | 39 | 38 | −1 |

Features with `can_mask=False` (RelativeChain, RelativeEntity,
RelativeSequenceSeparation, RelativeTokenSeparation, MSAOneHot) are correct.

`MSADataSource` has `can_mask=True` and `num_classes=6`, but the TorchScript
also uses `one_hot(., 6)` — the "mask" value is already encoded within the 6
classes by the generator itself (replacing with `NONE`). The port happens to
produce the right width here, but only by coincidence.

**Cumulative misalignment per feature group:**

- TOKEN: −4 channels (33+2560+2+1+33+1+33+3+4 = 2638 expected; port produces
  32+2560+1+1+33+1+32+2+3 = 2665? No — let me be precise). Actually the total
  deficit is 1+1+1+1 = 4 across ResidueType, IsDistillation, TokenBFactor,
  TokenPLDDT. Port produces 2634, zero-pads to 2638, but the last 4 channels
  are zeros instead of being positioned correctly throughout the feature vector.
- ATOM: −5 channels (4 from AtomNameOneHot, 1 from AtomRefElement).
- ATOM_PAIR: −1 channel.
- TOKEN_PAIR: −1 channel (DockingConstraintGenerator) + RBF issues (see below).
- MSA: 0 (MSADataSource is OK by coincidence, MSAOneHot has can_mask=False).
- TEMPLATES: −1 channel (TemplateDistogram).

**Required fix:** For features with `can_mask=True`, use
`one_hot(idx, num_classes + 1)`.  Remove the `clamp(0, num_classes - 1)` —
the mask index (`num_classes`) must be preserved and mapped to the extra
channel.  For `MSADataSource` specifically, this would create a 7th channel
that is always zero (since the generator replaces masks with `NONE` internally),
changing the dimension from the TorchScript's 6.  So the fix must be: use
`one_hot(idx, num_classes + 1)` only when `can_mask=True` AND the TorchScript
actually uses the extra channel.  Practically, this means applying the +1
rule universally for `can_mask=True` features except `MSADataSource`
(special-cased, since its `num_classes` already includes the mask class).

---

## Issue 2 — CRITICAL: AtomNameOneHot shape mismatch (mult=4)

**Files:** `featurize.py:_encode_one_hot`, `atom_name.py`.

`AtomNameOneHot` has `mult=4` — each atom has 4 character indices, producing a
`(B, N_atoms, 4)` raw tensor.  The port's `_encode_one_hot` does
`feat.squeeze(-1)` (no-op since last dim is 4, not 1), then
`F.one_hot((B, N, 4), 64)` → `(B, N, 4, 64)`.  This 4D tensor cannot be
concatenated along `dim=-1` with the 3D `(B, N, d)` tensors from other ATOM
features.

The TorchScript handles this correctly:

```python
_5 = torch.one_hot(AtomNameOneHot, 65)        # (B, N, 4, 65)
_4 = torch.reshape(embedded_feat, [B, N, 260]) # 4 × 65 = 260
```

**Required fix:** After one-hot encoding, reshape `(B, N, 4, num_classes+1)`
to `(B, N, 4*(num_classes+1))` — i.e., `(B, N, 260)`.  Detect this case via
`gen.mult > 1` or the feature name.

---

## Issue 3 — CRITICAL: RBF encoding not implemented

**Files:** `featurize.py:_encode_feature`, verified against
`feature_embedding_forward256.py` lines 202–238.

`TokenDistanceRestraint` and `TokenPairPocketRestraint` use `EncodingType.RBF`
with `num_rbf_radii=6`.  The port falls through to `_encode_identity`, passing
the raw 1-channel float through.  But the TorchScript performs RBF expansion
*inside* `feature_embedding.pt` using **learned radii** parameters:

```python
# TorchScript for TokenDistanceRestraint:
raw_data = unsqueeze(TokenDistanceRestraint, -1)
radii = self.feature_embeddings.TOKEN_PAIR.TokenDistanceRestraint.radii
diff = (radii - raw_data) / scale
encoding = exp(-clamp(diff², max=16))
encoding[clamped == 16] = 0  # kill gradient for far values
should_mask = (raw_data == -1.0).float()
encoding = encoding * (1 - should_mask)
result = cat([encoding, should_mask], dim=-1)  # → 7 channels
```

The learned `radii` parameter (6 values) and the `scale` constant are part of
the TorchScript module's state.  The port:
- Produces 1 channel instead of 7 for each RBF feature.
- Loses 12 channels total in TOKEN_PAIR (2 features × 6 deficit each).
- Has no parameter slots for `feature_embeddings.TOKEN_PAIR.TokenDistanceRestraint.radii`
  or `.TokenPairPocketRestraint.radii` — these learned weights are silently
  dropped during conversion.

Combined with Issue 1's TOKEN_PAIR deficit of 1 (DockingConstraintGenerator),
the total TOKEN_PAIR shortfall is **13 channels**: port produces 150, target
is 163, zero-padded at end.  All weights expecting features at positions
≥5 are misaligned.

**Required fix:** Either:
(a) Add the RBF encoding logic to `_batch_to_feature_context`, loading the
    learned `radii` from the TorchScript weights as fixed constants.  The
    `scale` constant appears from context to be derived from `num_rbf_radii`
    and `(max_dist - min_dist)` — verify against the TorchScript IR.
(b) Add `radii` parameters to `FeatureEmbedding` and implement the RBF
    encoding in `FeatureEmbedding.__call__`, updating the weight map in
    `name_map.py` to map
    `feature_embeddings.TOKEN_PAIR.{name}.radii → ...`.

Option (b) is more faithful and preserves the learned parameters.

---

## Issue 4 — CRITICAL: OUTERSUM TemplateResType uses learned Embedding, not one-hot

**Files:** `featurize.py:_encode_outersum`, `templates.py:TemplateResTypeGenerator`,
verified against `feature_embedding_forward256.py` lines 295–331.

The port's `_encode_outersum` does:

```python
oh = F.one_hot(idx, num_classes).float()
return oh[..., :, None, :] + oh[..., None, :, :]
```

This produces `num_classes` channels (~33) of one-hot outer sum.  But the
TorchScript uses a **learned `nn.Embedding`** with `embed_dim=32`:

```python
# TorchScript TemplateResType encoding:
offsets = arange(0, N_per_template) * num_classes
input4 = template_restype + offsets
embedded = embedding(input4)     # → (B, T, N, 1, 32)
row = reshape(embedded, [B, T, N, 1, 32])
col = reshape(embedded, [B, T, 1, N, 32])
emb = row + col                  # → (B, T, N, N, 32) broadcast
```

The outer-sum pattern is preserved (row + col), but applied to *learned 32-dim
embeddings*, not raw one-hot vectors.  This means:

1. **Wrong dimension**: port produces ~33 channels, TorchScript produces 32.
2. **Wrong representation**: one-hot outer sum vs learned embedding outer sum.
3. **Missing learned weight**: `feature_embeddings.TEMPLATES.TemplateResType.embedding.weight`
   has no corresponding parameter in the port's `FeatureEmbedding`.

The total TEMPLATES dimension: port would produce 39+2+33+3 = 77 instead of
39+2+32+3 = 76.  The port's `_concat_for_type(FeatureType.TEMPLATES, 76)`
would raise a RuntimeError because 77 > 76.

**Required fix:** Replace `_encode_outersum` for TemplateResType with an
embedding-based outer sum.  Add the embedding weight to either
`FeatureEmbedding` or as a standalone parameter, and update `name_map.py` to
map `feature_embeddings.TEMPLATES.TemplateResType.embedding.weight`.

---

## Issue 5 — MODERATE: Weight map drops 3 learned parameters

**Files:** `weights/name_map.py:_feature_embedding_map`.

The `_feature_embedding_map()` only maps the 6 `input_projs.*.0` Linear layers.
Three learned parameters in the TorchScript `feature_embedding.pt` are silently
dropped:

| TorchScript key | Purpose | Shape |
|----------------|---------|-------|
| `feature_embeddings.TEMPLATES.TemplateResType.embedding.weight` | Learned embedding for template res type outer sum | `(num_classes × N_per_template, 32)` |
| `feature_embeddings.TOKEN_PAIR.TokenDistanceRestraint.radii` | Learned RBF radii for distance restraint | `(6,)` |
| `feature_embeddings.TOKEN_PAIR.TokenPairPocketRestraint.radii` | Learned RBF radii for pocket restraint | `(6,)` |

The `convert_to_safetensors.py` script's `rename_state_dict` would place these
under `__unmapped__.*` and log a warning, but they'd never reach the model.

**Required fix:** Add these 3 parameters to the port's model (either in
`FeatureEmbedding` or as named buffers in `_batch_to_feature_context`'s caller)
and extend `_feature_embedding_map()` accordingly.

---

## Issue 6 — MODERATE: Parity validation bypasses the encoding bug

**Files:** `examples/validate_parity.py:validate_feature_embedding`.

The parity test generates random pre-encoded feature tensors (matching the
expected dimensions 2638, 163, etc.) and feeds them directly to the Linear
projections.  This validates the Linears but **never tests the encoding path**
(`_batch_to_feature_context`).  The bugs in Issues 1–4 would not be detected.

**Required fix:** Add a parity test that:
1. Creates an `AllAtomFeatureContext` using chai-lab.
2. Runs `Collate` + `feature_factory.generate`.
3. Runs the TorchScript `feature_embedding.pt` on the raw features.
4. Runs `_batch_to_feature_context` + `FeatureEmbedding` on the same raw features.
5. Compares outputs.

---

## Issue 7 — MODERATE: `run_pipeline.py` bypasses featurize_fasta entirely

**Files:** `examples/run_pipeline.py`.

The example creates dummy random tensors at the pre-encoded dimensions and
calls `featurize()` (the passthrough path), never exercising `featurize_fasta()`
or `_batch_to_feature_context`.  There is no integration test that runs the
full FASTA → features → embedding → trunk pipeline.

**Required fix:** Add an integration example/test that calls `featurize_fasta`
on a real or synthetic FASTA file and verifies the output dimensions match
the config.

---

## Issue 8 — MINOR: `_concat_for_type` concatenation order may not match TorchScript

**Files:** `featurize.py:_concat_for_type`.

Features are concatenated in `dict.items()` order (insertion order in Python
3.7+), which inherits from `feature_generators.items()` in `chai1.py`.  The
TorchScript concatenation order is hard-coded per feature type.

From `feature_embedding_forward256.py`:

**TOKEN** (line 156): ChainIsCropped, ESMEmbeddings, IsDistillation,
MSADeletionMean, MSAProfile, MissingChainContact, ResidueType, TokenBFactor,
TokenPLDDT — **alphabetical by feature name**.

**TOKEN_PAIR** (line 238): DockingConstraintGenerator, RelativeChain,
RelativeEntity, RelativeSequenceSeparation, RelativeTokenSeparation,
TokenDistanceRestraint, TokenPairPocketRestraint — **alphabetical**.

**MSA** (line 275): IsPairedMSA, MSADataSource, MSADeletionValue,
MSAHasDeletion, MSAOneHot — **alphabetical**.

The `feature_generators` dict in `chai1.py` uses insertion order, which is
NOT alphabetical (e.g., RelativeSequenceSeparation comes before
RelativeTokenSeparation, but ResidueType comes before ESMEmbeddings).

The port's `_concat_for_type` iterates over `groups.get(ft, [])` which
preserves the dict insertion order from `feature_generators`.  If this order
differs from the TorchScript's alphabetical order, features will be
misaligned even if individually correct.

**Required fix:** Verify that the TorchScript concatenation order matches the
Python dict iteration order, OR sort features alphabetically by name within
each type group before concatenation.  The TorchScript appears to use
alphabetical order throughout — sort in `_concat_for_type`.

---

## Issue 9 — MINOR: bond_adjacency double-sourcing

**Files:** `types.py:StructureInputs`, `featurize.py:_batch_to_feature_context`,
`embeddings.py:InputEmbedder`.

`bond_adjacency` appears in both `FeatureContext.bond_adjacency` and
`StructureInputs.bond_adjacency`.  In `InputEmbedder.__call__`:

```python
bond_adjacency = ctx.bond_adjacency
if bond_adjacency is None:
    bond_adjacency = ctx.structure_inputs.bond_adjacency
```

And `_batch_to_feature_context` sets `bond_adjacency=_mx(bond_features)` on
`FeatureContext` and also has `StructureInputs.bond_adjacency` as an
optional field.  This dual-path is not wrong but fragile — if both are set
with different values, the FeatureContext-level one wins silently.

**Required fix:** Pick one canonical location for bond adjacency and remove
the other, or document the precedence clearly.

---

## Issue 10 — MINOR: Missing msa_mask and template_input_masks in StructureInputs

**Files:** `types.py:StructureInputs`, `chai1.py` (lines 663–665).

The original pipeline computes:

```python
msa_mask = inputs["msa_mask"]
template_input_masks = und_self(inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2")
```

These masks are needed for the trunk.  The port's `StructureInputs` does not
carry `msa_mask` or `template_mask` — they would need to be passed separately
or stored in a different container.  The `Trunk.__call__` method presumably
receives them through `EmbeddingOutputs` or another mechanism, but the
`featurize_fasta` path does not appear to populate these masks.

**Required fix:** Ensure `featurize_fasta` / `_batch_to_feature_context`
extracts and passes `msa_mask` and `template_input_masks` through the pipeline
to the trunk.

---

## Efficiency Assessment

### What works well

1. **Delegation to chai-lab** is the right strategy — reimplementing 30+ feature
   generators would be error-prone and wasteful given they only run once per
   input (CPU-bound pre-processing).

2. **Dense pre-encoded tensors** fed to simple `nn.Linear` projections is
   efficient for MLX's lazy evaluation model — a single matmul per feature
   type replaces the TorchScript's per-feature dispatch.

3. **TokenBondRestraint** handled separately (as in the original) avoids
   complicating the feature_embedding export boundary.

4. **Weight conversion pipeline** (NPZ → rename → reshape → safetensors) is
   clean and well-documented.

### Efficiency concerns

1. **Torch→MLX conversion** in `_batch_to_feature_context` creates numpy
   intermediaries.  For large inputs (e.g., 2048 tokens), the template features
   tensor alone is `(1, 4, 2048, 2048, 76)` ≈ 4.8 GB in float32.  This could
   be streamed or converted in chunks.

2. **Dense one-hot encoding** materializes sparse categorical features into
   dense float tensors before the Linear projection.  This is computationally
   wasteful vs. the TorchScript's approach of `nn.Embedding` lookups (for
   OUTERSUM) or fused one-hot-then-matmul.  For the TOKEN features (2638-wide
   input), the one-hot encoding creates a ~2.5 MB tensor per token; for 2048
   tokens that's 5 GB just for the encoded token features.  An `nn.Embedding`
   approach would be much more memory-efficient.

3. **Template feature pair-expansion** creates `(B, T, N, N, 76)` tensors
   which for N=2048 is enormous.  The TorchScript avoids this by encoding
   on-the-fly.

---

## Summary of Required Fixes

| # | Severity | Issue | Impact | Status |
|---|----------|-------|--------|--------|
| 1 | CRITICAL | ONE_HOT mask channel missing for `can_mask=True` features | All feature groups misaligned; model produces garbage | **FIXED** — `_ONE_HOT_WIDTH` lookup table from TorchScript IR |
| 2 | CRITICAL | AtomNameOneHot `mult=4` not flattened after one-hot | Runtime shape error or silent misalignment | **FIXED** — reshape to `(B, N, 260)` when `result.ndim > feat.ndim` |
| 3 | CRITICAL | RBF encoding not implemented; 2 learned params dropped | TOKEN_PAIR 13 channels short; restraint features ignored | **FIXED** — `FeatureEmbedding._encode_rbf` with learned radii |
| 4 | CRITICAL | TemplateResType OUTERSUM uses one-hot instead of learned Embedding | TEMPLATES dimension wrong (77 vs 76); runtime error | **FIXED** — `FeatureEmbedding._encode_template_restype` with `nn.Embedding(33,32)` |
| 5 | MODERATE | Weight map drops 3 learned feature_embedding parameters | Model loads without error but encoding is numerically wrong | **FIXED** — 3 entries added to `_feature_embedding_map` |
| 6 | MODERATE | Parity validation never tests the encoding path | Bugs 1–4 would not be caught by existing tests | OPEN — `validate_parity.py` attribute name bug fixed; encoding-path parity test still needed |
| 7 | MODERATE | No integration test for featurize_fasta | Full pipeline never exercised | **FIXED** — `test_featurize_fasta.py` added (dimension checks) |
| 8 | MINOR | Feature concatenation order may not match TorchScript | Potential silent misalignment if order differs | **FIXED** — alphabetical sort in `_concat_for_type` |
| 9 | MINOR | bond_adjacency dual-sourcing | Fragile, could cause silent data loss | **FIXED** — documented `FeatureContext.bond_adjacency` as canonical, `StructureInputs` as fallback |
| 10 | MINOR | Missing msa_mask/template_mask in StructureInputs | Trunk may not receive required masks via featurize_fasta | **FIXED** (previously) — masks populated in `_batch_to_feature_context` and threaded through trunk |
