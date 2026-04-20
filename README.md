# chai-mlx

`chai-mlx` is an [MLX](https://github.com/ml-explore/mlx) inference port of
[Chai-1](https://github.com/chaidiscovery/chai-lab), the all-atom biomolecular
structure prediction model from Chai Discovery. It targets Apple silicon
(Metal) and ships with:

- a clean staged pipeline API (`ChaiMLX.embed_inputs` → `trunk` →
  `prepare_diffusion_cache` → `diffusion_step` → `confidence` →
  `rank_outputs`) plus a single-call `run_inference` wrapper;
- a FASTA-driven command-line runner (`chai-mlx-infer`) that writes
  chai-lab-compatible CIFs, per-sample score NPZs, and a run manifest;
- a set of CUDA comparison harnesses that run the chai-lab reference
  on [Modal](https://modal.com) H100s, plus local drivers that diff MLX
  vs CUDA tensor-for-tensor, structure-for-structure, and wall-clock;
- pre-converted model weights on
  [Hugging Face](https://huggingface.co/josephjojoe/chai-mlx).

No retraining, no numerical modification of the model — just the upstream
Chai-1 architecture and weights re-expressed in MLX, plus the tooling it
took to prove that the port is actually faithful.

## Status

- Structurally faithful end-to-end. The diffusion sampler is bit-for-bit
  exact given correct trunk outputs.
- On 1L2Y (Trp-cage, 20 residues), MLX vs CUDA (H100, bf16) Cα RMSD is
  **0.75 Å mean** across 15 sample pairs (3 seeds × 5 diffusion samples),
  vs **0.57 Å** for CUDA against the NMR ground truth. MLX sits ~0.26 Å
  further from experimental truth than CUDA on average, with
  **GDT-TS = 95.1%** and **Cα lDDT = 89.8%** between the two implementations.
- The remaining gap is dominated by TorchScript kernel-fusion differences
  in the chai-lab scripted `trunk.pt`, not by the MLX port. An eager
  PyTorch reimplementation of the exact same chai-lab module tree (same
  weights, same layout) at bf16 is 19.7% away from scripted-CUDA on the
  trunk pair tensor and 44% away on the single tensor, while MLX-fp32
  sits inside that envelope (14.9% and 42.2% from scripted-CUDA on the
  same tensors). Both eager-CUDA at fp32 and bf16 land in essentially
  the same place as MLX (1.3% between each other), and MLX → eager-CUDA
  is 12% on the full 48-block trunk at bf16. The full attribution — with
  per-round MSA-module intermediates, per-block pairformer intermediates,
  isolated tri-attention drift, and SDPA-variant analysis — lives in
  [`findings/drift_attribution.md`](findings/drift_attribution.md).
  Running MLX at `compute_dtype="float32"` does not meaningfully shrink
  the gap against scripted-CUDA, because scripted-CUDA itself is the
  outlier relative to both eager paths.

Numerically validated so far: monomers up to 76 residues
(1L2Y, 1VII, 1CRN, 1UBQ). The expanded validation slate (multimer,
ligand, >200 residue, nucleic acid, ESM, constraints) is wired up
end-to-end through the harnesses; the Modal sweep that populates its
numbers has not been run against the current checkpoint yet. See
[Validation coverage](#validation-coverage) for the target matrix and
how to reproduce each axis.

## Requirements

- Python ≥ 3.11
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.16 (Apple silicon only —
  CPU fallback is functional but orders of magnitude slower; MLX itself
  will warn about this)
- ~1.2 GB of disk for weights; 8 GB of unified memory comfortably fits
  small monomers. ESM-2 3B in-process adds ~11 GB — see
  [ESM-2 embeddings](#esm-2-embeddings).

## Install

```bash
git clone --recurse-submodules https://github.com/josephjojoe/chai-mlx
cd chai-mlx
pip install -e .
```

The `--recurse-submodules` flag fetches two pinned checkouts:

- `chai-lab/` — used by the featurizer and the CUDA comparison harnesses.
  The pin is a specific post-0.6.1 commit (see `pyproject.toml`); bare
  PyPI `chai_lab` drifts from the API this codebase expects.
- `esm-mlx/` — a separate MLX port of ESM-2 3B, used by
  `esm_backend="mlx"` / `"mlx_cache"` to compute language-model embeddings
  locally on Apple silicon.

If you cloned without `--recurse-submodules`, run
`git submodule update --init --recursive`.

### Optional extras

```bash
pip install -e ".[featurize]"     # torch + pinned chai_lab; required for FASTA featurization
pip install -e ".[esm]"           # esm-mlx; enables esm_backend="mlx" / "mlx_cache"
pip install -e ".[convert]"       # torch + safetensors; for TorchScript -> safetensors export
pip install -e ".[cuda-harness]"  # modal + gemmi + biopython; for the CUDA comparison harnesses
pip install -e ".[test]"          # pytest
```

### Installed console scripts

`pyproject.toml` exposes six console scripts; the legacy `scripts/*.py`
forwarders still work from a clone for backward compatibility.

| Console script                   | Purpose                                                         |
| -------------------------------- | --------------------------------------------------------------- |
| `chai-mlx-infer`                 | FASTA → CIFs + scores + manifest (`chai_mlx.cli.infer`)         |
| `chai-mlx-precompute-esm`        | Pre-compute ESM-MLX embeddings into a shared cache directory    |
| `chai-mlx-sweep`                 | Subprocess-per-target MLX sweep over the validation slate       |
| `chai-mlx-export-torchscript`    | Dump intermediate tensors from the TorchScript reference        |
| `chai-mlx-convert-torchscript`   | Convert chai-1 `.pt` checkpoints → MLX safetensors              |
| `chai-mlx-convert-npz`           | Convert legacy per-component NPZ weight dumps → safetensors     |

## Quick start

### Python API

```python
from chai_mlx import ChaiMLX, featurize_fasta

# Pulls ~1.2 GB of safetensors from the HF repo on first call.
# Cached in the standard HF cache on subsequent calls.
model = ChaiMLX.from_pretrained("josephjojoe/chai-mlx")

ctx = featurize_fasta("input.fasta", output_dir="./out")  # needs [featurize] extra
result = model.run_inference(ctx, recycles=3, num_samples=5, num_steps=200)
# result.coords:     MLX array, shape (B, S, A, 3)
# result.confidence: pae_logits, pde_logits, plddt_logits
# result.ranking:    aggregate_score, ptm, iptm, per-chain breakdowns, clashes
```

`ChaiMLX.from_pretrained(...)` accepts either a Hugging Face repo id (as
above, via `huggingface_hub`) or a local directory containing
`config.json` plus `model.safetensors` (or sharded safetensors with an
index file). Pass `compute_dtype="float32"` to disable mixed-precision
inference.

`model.run_inference_debug(...)` returns a superset `FoldOutputs` that
retains the feature context, embeddings, and trunk intermediates
alongside coordinates / confidence / ranking — useful when feeding
individual tensors to the CUDA comparison harnesses.

For a no-weights smoke test of the full pipeline on random inputs, see
`examples/basic_inference.py`.

### Command line

```bash
chai-mlx-infer \
    --weights-dir josephjojoe/chai-mlx \
    --fasta path/to/input.fasta \
    --output-dir ./out
```

This writes (per trunk sample, flat if `--num-trunk-samples=1`):

```text
out/
  input.fasta              copy of the input FASTA (for reference)
  pred.model_idx_{0..N}.cif   one CIF per diffusion sample; per-atom
                              pLDDT is written as the B-factor column
  scores.json              aggregate_score / ptm / iptm per sample
  scores.model_idx_{0..N}.npz  chai-lab-compatible per-sample npz with
                              pae / pde / plddt per-token tensors
  manifest.json            dtype, recycles, steps, wall clock, all flags
  _features/               intermediate featurizer artifacts (MSAs, etc.)
```

Common flags:

- `--num-samples 5` — diffusion samples per trunk sample (chai-1 default)
- `--num-trunk-samples 1` — independent trunk runs at seeds
  `{seed, seed+1, …}`; when >1, outputs land under
  `out/trunk_{i}/` with the same layout as above
- `--recycles 3 --num-steps 200` — chai-1 defaults
- `--dtype bfloat16|float32` — mixed precision vs full-precision
- `--constraint-path restraints.csv` — chai-lab contact + pocket +
  covalent-bond restraints
- `--use-msa-server` / `--msa-directory <dir>` — online ColabFold vs
  offline cached MSAs; when a prior MSA cache is present, it's reused
  automatically (pass `--refresh-msa` to force re-fetching)
- `--esm-backend {off,chai,mlx,mlx_cache} [--esm-cache-dir <dir>]` —
  ESM-2 embeddings; see [ESM-2 embeddings](#esm-2-embeddings) below
- `--fasta-dir <dir>` — batch mode: one `*.fasta` per subdirectory
  under `--output-dir`, plus a top-level `run_summary.json`
- `--write-msa-plot` — emit `msa_coverage.png` when an MSA source is
  active

Run `chai-mlx-infer --help` for the exhaustive list.

### FASTA format

chai-lab's `>kind|name=SHORT` header grammar is enforced up front by
`chai_mlx.data.fasta.validate_fasta_or_raise`, so malformed inputs fail
before the model loads:

```text
>protein|name=A
MKFLILFNILVSTLSFSSAQA
>ligand|name=LIG
CC(=O)Oc1ccccc1C(=O)O
>dna|name=D1
ACGTACGTACGT
```

`kind` must be one of `protein`, `ligand`, `dna`, `rna`, `glycan`;
`SHORT` ≤ 4 characters (chai-lab's fixed-length tensor encoding).

## ESM-2 embeddings

`featurize_fasta` / `chai-mlx-infer` take an `esm_backend` knob that
selects how the 3 B-parameter ESM-2 per-residue embeddings are obtained:

| Backend       | What happens                                                                           | When to use                                                              |
| ------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `off`         | ESM feature is zero-filled                                                             | Default for CUDA parity runs; MLX-only tests                             |
| `chai`        | chai-lab's traced CUDA fp16 checkpoint                                                 | Only on CUDA hosts; what the reference uses                              |
| `mlx`         | Run `esm-mlx` in-process (loads ~11 GB of weights)                                     | Convenient on Macs with ≥ 32 GB unified memory                           |
| `mlx_cache`   | Load pre-computed `<sha1(seq)>.npy` files from a directory                             | Recommended for 16 GB Macs — no extra RAM cost at inference time         |

The `mlx_cache` workflow is two-step: pre-compute once, then run
inference with the cache attached so the chai-mlx Python process never
holds ESM weights.

```bash
# 1. Pre-compute ESM-MLX embeddings for every protein record in a FASTA
#    (requires [esm] extra).  Writes <sha1>.npy per unique sequence.
chai-mlx-precompute-esm \
    --fasta path/to/input.fasta \
    --cache-dir ./esm_cache

# 2. Run inference against the cache (zero extra RAM cost).
chai-mlx-infer \
    --weights-dir josephjojoe/chai-mlx \
    --fasta path/to/input.fasta \
    --output-dir ./out \
    --esm-backend mlx_cache \
    --esm-cache-dir ./esm_cache
```

## Weights

Pre-converted weights are hosted on Hugging Face at
[`josephjojoe/chai-mlx`](https://huggingface.co/josephjojoe/chai-mlx)
(~1.2 GB, float32 safetensors, sharded with an index file). They are
Chai Discovery's released Chai-1 weights, re-expressed in MLX's naming
convention — no retraining or numerical modification.

If you'd rather start from the upstream TorchScript distribution, fetch
the `.pt` files from Chai's CDN and convert locally:

```bash
for f in trunk token_embedder feature_embedding diffusion_module \
         confidence_head bond_loss_input_proj; do
  curl -O "https://chaiassets.com/chai1-inference-depencencies/models_v2/${f}.pt"
done

chai-mlx-convert-torchscript --pt-dir . --out-dir weights/
```

Alternatively, with a Modal account set up,
`cuda_harness.modal_common::download_inference_dependencies` primes a
shared Modal Volume with the same files (the CUDA comparison harnesses
read from it).

The on-disk schema (`weights/config.json` plus per-component safetensors)
is versioned via `ChaiConfig.config_version`; mismatched versions print
a loud warning before attempting to load.

## Repository layout

```text
chai_mlx/             MLX package
  __init__.py         re-exports ChaiMLX, ChaiConfig, featurize, featurize_fasta, …
  config.py           architecture/config dataclasses (ChaiConfig + sub-configs)
  utils.py            numerics, masking, geometry, schedule helpers
  cli/                installed console-script entry points
    infer.py          chai-mlx-infer — FASTA → CIFs + scores + manifest
    precompute_esm_impl.py  chai-mlx-precompute-esm — ESM cache population
    sweep_impl.py     chai-mlx-sweep — subprocess-per-target MLX sweep
  data/               featurization adapters and typed contexts
    featurize.py      featurize() (precomputed tensors) + featurize_fasta()
    fasta.py          lightweight chai-lab-format FASTA parser + validators
    types.py          FeatureContext, StructureInputs, outputs, etc.
    esm_mlx_adapter.py  esm-mlx integration; sha1-keyed in-memory cache
    _rdkit_timeout_patch.py  macOS RDKit timeout workaround
  model/              public pipeline (ChaiMLX) and the five major stages
    core.py           ChaiMLX, run_inference, run_inference_debug, from_pretrained
    embeddings.py     FeatureEmbedding + InputEmbedder
    trunk.py          Pairformer + MSA/template modules (48 blocks)
    diffusion.py      EDM schedule, atom attention, denoiser
    confidence.py     pLDDT/PAE/PDE head (4 pairformer blocks)
    ranking.py        ranking (pTM/ipTM/clashes/aggregate)
  nn/                 reusable neural-network building blocks
    layers/           attention, atom attention, pairformer, triangle, common
  io/                 IO / persistence
    weights/          load.py, export_torchscript.py, convert_torchscript.py,
                      convert_npz.py, name_map.py, validate.py

cuda_harness/         chai-lab-on-CUDA reference harness (Modal app)
  modal_common.py     shared app / image / weights volume / DEFAULT_TARGETS
  run_reference.py    end-to-end CUDA inference per (target, seed)
  run_intermediates.py same, plus per-boundary tensor dumps
  run_determinism.py  two CUDA runs at the same seed; precision policy
  run_expanded_targets.py  CUDA reference over the expanded validation slate
  bench_throughput.py per-module CUDA wall clock
  smoke_test.py       minimal "does Modal still work?" check
  constraints/        chai-lab constraint CSVs used by expanded targets
  _probe_*.py         one-off stage-isolation probes (for debugging)

scripts/              contributor tooling
  inference.py        forwarder to chai_mlx.cli.infer
  precompute_esm_mlx.py  forwarder to chai_mlx.cli.precompute_esm_impl
  run_mlx_sweep.py    forwarder to chai_mlx.cli.sweep_impl
  cuda_parity.py      stage-by-stage MLX-vs-CUDA diff from an intermediates NPZ
  cuda_structure_sweep.py  MLX vs CUDA on final 3D structures (RMSD/GDT/lDDT)
  cuda_error_accumulation.py  how MLX-vs-CUDA grows across recycles + steps
  cuda_determinism_report.py  attributes gap to CUDA non-determinism
  cuda_mlx_diffusion_isolation.py  trunk drift vs MLX diffusion sampler
  cuda_constraints_parity.py  constraint featurizer MLX vs CUDA parity
  mlx_throughput.py + report_throughput_comparison.py  MLX vs CUDA wall clock
  compare_vs_pdb.py + summarise_vs_pdb.py  Kabsch-aligned vs experimental PDB
  run_t2_parity.py    T=2 micro-parity probe
  spawn_cuda_sweep.py fan every CUDA experiment out to Modal in parallel

examples/             minimal runnable examples
  basic_inference.py  random-input smoke (no weights, no featurizer)
  fasta_smoke.py      FASTA featurization + dim check
  diffusion_benchmark.py  diffusion-loop wall clock

tests/                pytest suite (33 files; ~3 s for the default collection)
chai-lab/             pinned chai-lab checkout (git submodule)
esm-mlx/              pinned esm-mlx checkout (git submodule)
weights/              local model artifacts (gitignored)
auxiliary/            reference-only preprint excerpts and diagrams
findings/             graph dumps from the TorchScript reference
LICENSE, NOTICE       Apache-2.0 + upstream attribution
```

## Local workflows

- **Smoke the package on random inputs** (no weights, no featurizer):
  `python examples/basic_inference.py`
- **Smoke the featurizer** (needs `[featurize]`):
  `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- **End-to-end FASTA inference** (needs `[featurize]`):
  `chai-mlx-infer --weights-dir josephjojoe/chai-mlx \
     --fasta path/to/input.fasta --output-dir ./out`
- **Benchmark the diffusion loop**:
  `python examples/diffusion_benchmark.py`
- **Run the tests**:
  `pip install -e ".[test]"` then `pytest -q`
  (the slow end-to-end tests that download HF weights are opt-in via
  `CHAI_MLX_RUN_SLOW=1 pytest -q -m slow`)

## CUDA comparison harnesses

The chai-lab CUDA reference cannot run end-to-end on a 16 GB Mac — the
upstream TorchScript stack is memory-unoptimised. Instead, everything
in `cuda_harness/` runs on [Modal](https://modal.com) on H100s and
produces artifacts that local `scripts/cuda_*` helpers consume. No
local GPU needed.

### Prerequisites

1. `pip install -e ".[cuda-harness]"`
2. A Modal account configured via `modal setup`
   (confirm with `modal profile current`).
3. One-time weight cache on the Modal Volume:

   ```bash
   modal run -m cuda_harness.modal_common::download_inference_dependencies
   ```

### Harnesses (CUDA side)

| Harness                                      | What it does                                                                                                                                        |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cuda_harness.run_reference`                 | End-to-end chai-lab CUDA inference per (target, seed); emits CIFs + score NPZ per diffusion sample.                                                 |
| `cuda_harness.run_intermediates`             | Same flow, plus per-boundary tensor dumps (embedding, bond projection, per-recycle trunk, diffusion snapshots, confidence logits, ranking) per seed. |
| `cuda_harness.run_determinism`               | Runs chai-lab twice at the same seed under configurable precision policy (`default` / `tf32_off` / `deterministic`) to measure CUDA non-determinism. |
| `cuda_harness.run_expanded_targets`          | CUDA reference over the expanded validation slate (multimer/ligand/long/dna/esm/constraints) with `use_esm_embeddings=True`.                         |
| `cuda_harness.bench_throughput`              | Per-module CUDA wall clock with warmup + `cuda.synchronize` gates; per-target JSON + combined CSV.                                                  |
| `cuda_harness.smoke_test`                    | Minimal "does the Modal app still work?" canary.                                                                                                    |

Invoke any of them as `modal run -m cuda_harness.<harness>`.

### Harnesses (local, MLX side)

| Script                                          | Question it answers                                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `scripts/cuda_parity.py`                        | Does MLX match CUDA tensor-for-tensor, stage by stage?                                                       |
| `scripts/cuda_structure_sweep.py`               | How much do MLX and CUDA diverge on final 3D structures (RMSD / GDT / lDDT / ranking scores), optionally vs experimental PDB? |
| `scripts/cuda_error_accumulation.py`            | How does MLX-vs-CUDA error grow across trunk recycles and diffusion steps?                                   |
| `scripts/cuda_determinism_report.py`            | How much of the MLX-vs-CUDA gap is just CUDA disagreeing with itself?                                        |
| `scripts/cuda_mlx_diffusion_isolation.py`       | How much of the structural gap is trunk drift vs MLX diffusion sampler differences?                          |
| `scripts/cuda_constraints_parity.py`            | Do MLX and CUDA produce identical constraint features given the same CSV?                                    |
| `scripts/mlx_throughput.py` + `report_throughput_comparison.py` | Side-by-side MLX vs CUDA per-module wall clock.                                               |
| `scripts/compare_vs_pdb.py` + `summarise_vs_pdb.py`              | Kabsch-aligned MLX/CUDA vs experimental PDB; handles synthetic target names, multimers, nucleic-acid P-backbones. |
| `scripts/spawn_cuda_sweep.py`                   | Fan every CUDA experiment out to Modal in parallel.                                                          |

### Example workflow

```bash
# 1. Numerical parity (single target + seed, full stage-isolation diff).
modal run -m cuda_harness.run_intermediates --targets 1L2Y --seeds 42

python scripts/cuda_parity.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \
    --summary-json /tmp/chai_mlx_cuda/parity_1L2Y_seed42.json

# 2. Structure-level agreement across targets and seeds.
modal run -m cuda_harness.run_reference \
    --targets 1L2Y,1VII,1CRN,1UBQ --seeds 0,42,123

python scripts/cuda_structure_sweep.py \
    --weights-dir weights \
    --reference-dir /tmp/chai_mlx_cuda/reference \
    --mlx-output-dir /tmp/chai_mlx_cuda/mlx \
    --compare-pdb --csv /tmp/chai_mlx_cuda/structure_sweep.csv

# 3. CUDA run-to-run determinism.
modal run -m cuda_harness.run_determinism \
    --targets 1L2Y --seeds 42 --precision default
python scripts/cuda_determinism_report.py \
    --npz /tmp/chai_mlx_cuda/determinism/1L2Y/seed_42_default.npz

# 4. Trunk-vs-diffusion attribution.
python scripts/cuda_mlx_diffusion_isolation.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \
    --cuda-reference-dir /tmp/chai_mlx_cuda/reference/1L2Y/seed_42

# 5. Throughput: CUDA on Modal, MLX locally, then a side-by-side report.
modal run -m cuda_harness.bench_throughput --targets 1L2Y,1VII,1CRN,1UBQ

python scripts/mlx_throughput.py \
    --weights-dir weights \
    --targets 1L2Y,1VII,1CRN,1UBQ \
    --output-json /tmp/chai_mlx_cuda/throughput/mlx.json

python scripts/report_throughput_comparison.py \
    --mlx-json /tmp/chai_mlx_cuda/throughput/mlx.json \
    --cuda-dir /tmp/chai_mlx_cuda/throughput
```

## Validation coverage

The target slate in `cuda_harness/modal_common.py::DEFAULT_TARGETS` is
the single source of truth for "what has chai-mlx been pointed at?".
Each target is tagged with one or more `kinds`, and harnesses accept a
`--target-kinds` filter so you can sweep a single axis at a time.

| Target        | Kind(s)              | Entities                                 | What it exercises                                        | MLX vs CUDA      |
| ------------- | -------------------- | ---------------------------------------- | -------------------------------------------------------- | ---------------- |
| `1L2Y`        | monomer              | 1 protein (20 aa)                        | Baseline Cα RMSD / GDT / lDDT                            | **0.75 Å**, 95.1% GDT |
| `1VII`        | monomer              | 1 protein (35 aa)                        | Small α-helical fold                                     | measured         |
| `1CRN`        | monomer              | 1 protein (46 aa)                        | Crambin, tight 2.0 Å PDB reference                       | measured         |
| `1UBQ`        | monomer              | 1 protein (76 aa)                        | Mid-size α/β monomer; previous ceiling                   | measured         |
| `1BRS`        | multimer             | 2 proteins (barnase + barstar, 199 aa)   | Multi-chain featurizer, interface Cα RMSD, iPTM          | scaffolded       |
| `1FKB`        | ligand               | FKBP-12 + FK506 SMILES                   | `EntityType.LIGAND` path, ligand heavy-atom RMSD         | scaffolded       |
| `7TIM`        | long                 | 1 protein (248 aa, TIM barrel)           | >200 residue ceiling, bf16 error growth                  | scaffolded       |
| `1BNA`        | dna, multimer        | 2 DNA strands (12 bp duplex)             | `EntityType.DNA` featurization, P-backbone RMSD          | scaffolded       |
| `1UBQ_ESM`    | esm                  | 1 protein (76 aa)                        | MLX ESM-2 3B vs chai-lab's traced CUDA checkpoint        | scaffolded       |
| `1CRN_CONSTR` | constraints, ligand  | 1 protein + methanethiol + 3 restraints  | Contact / pocket / covalent restraint features           | scaffolded       |

"measured" means numbers are in [Status](#status) above; "scaffolded"
means the plumbing is end-to-end ready but the sweep that populates
numbers has not yet run against the current checkpoint.

### Reproducing the expanded sweep

Assumes `[cuda-harness]` and `[featurize]` extras are installed and a
`modal` profile is set up. No MSA or template servers are used
(offline only). Local MLX inference uses the `[esm]` extra when
`--esm-backend mlx` is set.

```bash
# 1. CUDA reference runs for every target on one seed
#    (use_esm_embeddings=True on the Modal side).
modal run -m cuda_harness.run_expanded_targets --seeds 42

# 2. Local MLX + per-kind structural comparison.
python scripts/cuda_structure_sweep.py \
    --weights-dir weights \
    --reference-dir /tmp/chai_mlx_cuda/expanded \
    --mlx-output-dir /tmp/chai_mlx_cuda/mlx_expanded \
    --csv /tmp/chai_mlx_cuda/expanded_sweep.csv

# 3. Constraint-feature parity (MLX featurizer vs CUDA featurizer on the same CSV).
modal run -m cuda_harness.run_intermediates \
    --targets 1CRN_CONSTR --seeds 42 \
    --constraint-resource 1CRN_all_three.csv

python scripts/cuda_constraints_parity.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1CRN_CONSTR/seed_42.npz

# 4. ESM-on-MLX vs ESM-on-CUDA (one seed, one sample pair).
chai-mlx-infer --weights-dir weights/ \
    --fasta /tmp/chai_mlx_cuda/expanded/1UBQ_ESM/seed_42/input.fasta \
    --output-dir /tmp/chai_mlx_cuda/mlx_expanded/1UBQ_ESM \
    --esm-backend mlx --save-npz /tmp/chai_mlx_cuda/mlx_expanded/1UBQ_ESM.npz
```

Step 2's CSV is wide: per-row it carries per-chain Cα RMSDs, an
interface Cα RMSD for multimers, a heavy-atom RMSD for ligands, and a
phosphate-backbone RMSD for DNA/RNA, alongside the aggregate MLX-vs-CUDA
score gap. The per-kind summary printed at the end of the run is what
drops into this README once the sweep has been executed.

## Architecture

Chai-1 is an AlphaFold-3-class all-atom structure prediction model with
two principal stages:

1. A **trunk** that produces single- and pair-token representations from
   the input features (sequence, MSA, templates, constraints, ESM-2
   embeddings). The trunk is 48 blocks of pairformer
   (single/pair attention + triangle attention + transitions), with
   MSA outer-product-mean and pair-weighted-averaging blocks
   conditioning the pair representation. The trunk is recycled 3 times
   by default.
2. A **diffusion module** that denoises atom coordinates over 200 EDM
   steps, conditioned on the trunk outputs via atom attention. A
   separate confidence head (4 pairformer blocks) produces pLDDT /
   PAE / PDE logits from the final coordinates, and a ranker computes
   pTM / ipTM / clash counts / aggregate score to pick the best sample.

See `auxiliary/preprint-af3-modifications.md` for a summary of how
Chai-1 differs from AlphaFold-3 (language-model embeddings, constraint
features), and `findings/graphs/` for TorchScript IR dumps extracted
from the upstream reference checkpoint — useful when reverse-engineering
specific submodules.

The MLX port preserves the upstream architecture and weight layout
exactly. Every dimension is pinned in `chai_mlx/config.py::ChaiConfig`,
and the on-disk `config.json` that ships with the HF weights is
versioned via `ChaiConfig.config_version` so future schema changes fail
loudly rather than silently loading into incompatible modules.

## License

Apache-2.0. This project is a derivative work of
[chai-lab](https://github.com/chaidiscovery/chai-lab) (also Apache-2.0);
see `NOTICE` for attribution. Pre-converted MLX weights are derived
from Chai Discovery's released Chai-1 weights, distributed under
[their own terms](https://github.com/chaidiscovery/chai-lab). The
upstream chai-lab LICENSE is preserved in the pinned `chai-lab/`
submodule.
