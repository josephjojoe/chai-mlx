# chai-mlx

`chai-mlx` is an MLX port of [Chai-1](https://github.com/chaidiscovery/chai-lab),
the protein structure prediction model. It targets Apple silicon and ships
with a set of harnesses for comparing the MLX implementation against the
CUDA reference that actually runs in production.

## Status

- Structurally faithful end-to-end. Diffusion is bit-for-bit exact given
  correct trunk outputs.
- Validated on the full 10-target slate: monomer (1L2Y, 1VII, 1CRN,
  1UBQ), multimer (1BRS), ligand (1FKB), long monomer (7TIM), DNA
  (1BNA), ESM target (1UBQ_ESM), constraints (1CRN_CONSTR). With ESM
  on, MLX wins against the experimental PDB on 6/10 targets and CUDA
  wins on 4/10, with the deltas dominated by bf16 accumulation in the
  48-block pairformer rather than any structural bug — same well,
  different samples. Full numbers and the "collapse once ESM is on"
  MLX↔CUDA convergence are in `HANDOFF.md` §1.1 – §1.5.
- 40 offline tests (`pytest -q`) cover the featurizer (protein, DNA,
  multimer, constraints including covalent bonds, ligand end-to-end),
  attention / embeddings / trunk / diffusion / ranking, weight
  loading, the ESM-MLX adapter, and the `scripts/inference.py` CLI
  surface.

Plumbed but not yet exercised end-to-end in this repo: online MSA
(ColabFold API), online templates, offline MSA directories, offline
template m8 files, RNA-only inputs, glycan inputs. The code paths
call chai-lab's own server clients / offline loaders — i.e. they are
expected to work — but there are no regression numbers for them here.
See `HANDOFF.md` §1 and §8 for the caveat log. If you need one of
these paths, `scripts/inference.py` exposes the flags and is the
place to drive them.

## Install

```bash
git clone --recurse-submodules https://github.com/josephjojoe/chai-mlx
cd chai-mlx
pip install -e .
```

The `--recurse-submodules` flag fetches two pinned checkouts: `chai-lab/`
(used by the featurizer and comparison harnesses) and `esm-mlx/` (used
to produce ESM-2 embeddings locally on Apple silicon, avoiding the need
for torch + chai-lab's traced 3B CUDA checkpoint). If you've already
cloned without it, run `git submodule update --init --recursive`.

Optional extras:

```bash
pip install -e ".[featurize]"     # torch + chai_lab; required for FASTA featurization
pip install -e ".[esm]"           # esm-mlx; enables esm_backend="mlx" in featurize_fasta
pip install -e ".[convert]"       # torch + safetensors; for TorchScript -> safetensors export
pip install -e ".[cuda-harness]"  # modal + gemmi + biopython; for the CUDA comparison harnesses
pip install -e ".[test]"          # pytest
```

## Quick start

### Run chai-1 on a FASTA with one command

```bash
python scripts/inference.py \
    --weights-dir weights \
    --fasta input.fasta \
    --output-dir out/my_run
```

That's the main entry point. It handles any combination of protein,
ligand, DNA, RNA, and glycan chains that chai-1 itself supports, and
writes one mmCIF per diffusion sample plus a `scores.json`
(pTM / ipTM / pLDDT / aggregate) and a `manifest.json` (dtype, recycles,
steps, wall clock, input references) — the same polished output layout
the validation sweep produces. See [FASTA format](#fasta-format) below
for the header conventions and [CLI reference](#cli-reference) for
every flag.

Optional extensions covered by the same CLI:

```bash
# With contact / pocket / covalent-bond restraints (chai-lab CSV).
python scripts/inference.py --weights-dir weights --fasta input.fasta \
    --output-dir out/ --constraint-path constraints.csv

# With a pre-computed local ESM cache (recommended on Apple silicon).
python scripts/precompute_esm_mlx.py --cache-dir esm_cache --fasta input.fasta
python scripts/inference.py --weights-dir weights --fasta input.fasta \
    --output-dir out/ --esm-backend mlx_cache --esm-cache-dir esm_cache

# With online MSAs + templates (ColabFold API; plumbed, not validated
# here, see HANDOFF.md §1 for coverage caveats).
python scripts/inference.py --weights-dir weights --fasta input.fasta \
    --output-dir out/ --use-msa-server --use-templates-server

# With a pre-computed local MSA directory / template m8 file.
python scripts/inference.py --weights-dir weights --fasta input.fasta \
    --output-dir out/ --msa-directory my_msas/ --templates-path my_templates.m8
```

### Or call the Python API directly

```python
from chai_mlx import ChaiMLX, featurize_fasta

# Pulls ~1.2 GB of safetensors from the HF repo on first call.
model = ChaiMLX.from_pretrained("josephjojoe/chai-mlx")

ctx = featurize_fasta(
    "input.fasta",
    output_dir="./out",
    constraint_path=None,          # chai-lab restraint CSV, optional
    msa_directory=None,            # offline MSA dir, optional
    use_msa_server=False,          # ColabFold online MSA
    use_templates_server=False,    # online templates (requires use_msa_server=True)
    templates_path=None,           # offline templates m8, optional
    esm_backend="off",             # "off" | "chai" | "mlx" | "mlx_cache"
    esm_cache_dir=None,            # required when esm_backend="mlx_cache"
)
result = model.run_inference(ctx, recycles=3, num_samples=5, num_steps=200)
# result.coords, result.confidence, result.ranking
```

`ChaiMLX.from_pretrained(...)` accepts either a Hugging Face repo id (as
above, via `huggingface_hub`) or a local directory containing `config.json`
plus `model.safetensors` (or sharded safetensors with an index file). Pass
`compute_dtype="float32"` to disable mixed-precision inference.

For a no-weights smoke test of the pipeline, see
`examples/basic_inference.py`. For debug outputs (context + embeddings +
trunk intermediates alongside coords / confidence / ranking), use
`model.run_inference_debug(...)`.

The full set of staged entry points — `embed_inputs`, `trunk`,
`prepare_diffusion_cache`, `diffusion_step`, `confidence`, `rank_outputs` —
is documented in `chai_mlx/model/core.py`; they're what the CUDA comparison
harnesses call into, and are useful when you want to feed reference tensors
into individual stages.

### FASTA format

Chai-1 (and therefore `scripts/inference.py`) uses chai-lab's header
convention:

```text
>kind|name=SHORT
SEQUENCE
```

`kind` is one of `protein`, `ligand`, `dna`, `rna`, or `glycan`. `SHORT`
is a per-entity name of **at most 4 characters** (chai-lab packs it
into a fixed-length tensor when `entity_name_as_subchain=True`, which
is what the featurizer uses so constraint CSVs can address chains by
the same label the user wrote — see `HANDOFF.md` §5.4). `SEQUENCE` is
an amino-acid string for protein, a nucleotide string for dna / rna,
or a SMILES string for ligand. Multiple records in one file form a
complex; chai-1 will fold them together.

Example (protein + ligand):

```text
>protein|name=FKBP
GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLGKQEVIRGWE
EGVAQMSVGQRAKLTISPDYAYGATGHPGIIPPHATLVFDVELLKLE
>ligand|name=FK
CC=CC1CC(C)CC2(O)OC3(C(CC(C3O)C)C)OC(=O)C(N4CCCCC4C(=O)O2)
CC(=O)C(C)C=C(C)C(O)C(OC)CC(=O)C(C)CC(=C)C(C)C(O)C(C)C1OC
```

### CLI reference

`python scripts/inference.py --help` lists every flag. A condensed view:

| Group | Flag | Notes |
| --- | --- | --- |
| core | `--weights-dir` | Local dir with `config.json` + safetensors (or HF repo id). |
| core | `--fasta` | Input FASTA; required. |
| core | `--output-dir` | Where `pred.model_idx_*.cif`, `scores.json`, `manifest.json` land. |
| model | `--dtype` | `bfloat16` (default) or `float32`. |
| model | `--recycles` / `--num-steps` / `--num-samples` / `--seed` | Sampling knobs; defaults match the validation sweep (3 / 200 / 5 / 42). |
| constraints | `--constraint-path` | Chai-lab restraint CSV (contact + pocket + covalent). |
| MSA | `--msa-directory` | Offline MSA dir, mutually exclusive with `--use-msa-server`. |
| MSA | `--use-msa-server` / `--msa-server-url` | Online MSA via ColabFold. |
| templates | `--templates-path` | Offline templates m8 file. |
| templates | `--use-templates-server` | Online templates; requires `--use-msa-server`. |
| ESM | `--esm-backend` | `off` (default), `chai` (CUDA only), `mlx` (16 GB RAM), `mlx_cache`. |
| ESM | `--esm-cache-dir` | Required with `mlx_cache`; pre-computed `<sha1>.npy` directory. |
| output | `--save-npz` | Also dump raw coords + scores as npz (CIFs are always written). |
| output | `--skip-cif` | Skip CIF output (scores + optional npz only). |
| output | `--feature-dir` | Override where chai-lab caches MSA / template artifacts. |

Validation coverage for each knob is summarised in `HANDOFF.md` §1 —
what's been end-to-end measured vs what's plumbed-but-not-yet-exercised
in this repo.

## Weights

Pre-converted weights are hosted on Hugging Face at
[`josephjojoe/chai-mlx`](https://huggingface.co/josephjojoe/chai-mlx)
(~1.2 GB, float32 safetensors). These are Chai Discovery's released
Chai-1 weights, rewritten as safetensors for MLX — no retraining or
numerical modification.

If you'd rather start from the upstream TorchScript distribution, fetch
the `.pt` files from Chai Discovery's CDN and convert locally:

```bash
for f in trunk token_embedder feature_embedding diffusion_module \
         confidence_head bond_loss_input_proj; do
  curl -O "https://chaiassets.com/chai1-inference-depencencies/models_v2/${f}.pt"
done
chai-mlx-convert-torchscript --pt-dir . --out-dir weights/
```

Or, if you have a Modal account set up, `cuda_harness/modal_common.py`
exposes a `download_inference_dependencies` Function that populates a
Modal Volume with the same files (used by the CUDA harnesses below).

## Repository layout

```text
chai_mlx/           MLX package
  config.py         architecture/config dataclasses
  utils.py          numerics, masking, geometry, schedule helpers
  data/             featurization adapters and typed contexts
  model/            public model pipeline (ChaiMLX) and major stages
  nn/               reusable neural-network building blocks
  io/weights/       weight export, conversion, loading, validation

cuda_harness/       Modal-hosted chai-lab-on-CUDA reference harness
examples/           minimal runnable examples
scripts/            CUDA comparison drivers + FASTA inference runner
tests/              pytest suite (27 tests, ~3 s)
chai-lab/           pinned upstream reference checkout (git submodule)
weights/            local model artifacts (gitignored; see "Weights" above)
LICENSE, NOTICE     Apache-2.0 + upstream attribution
```

## Running locally

- **Smoke the package on random inputs** (no weights, no featurizer):
  `python examples/basic_inference.py`
- **Run end-to-end FASTA inference** (needs `[featurize]` extra):
  `python scripts/inference.py --weights-dir weights/ --fasta path/to/input.fasta --output-dir out/`
  → writes `pred.model_idx_*.cif` + `scores.json` + `manifest.json`.
  Full CLI is in [Quick start / CLI reference](#cli-reference).
- **Benchmark the diffusion loop**:
  `python examples/diffusion_benchmark.py`
- **FASTA featurization smoke**:
  `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- **Tests**: `pip install -e ".[test]"` then `pytest -q` (≈30 s with
  `[featurize]` installed; `test_ranking.py` parity tests are auto-
  skipped without it). See `HANDOFF.md` §1.8 for the test count.

## CUDA comparison harnesses

The CUDA reference cannot run end-to-end on a 16 GB Mac because the upstream
TorchScript stack is memory-unoptimised. Instead, everything in
`cuda_harness/` runs on [Modal](https://modal.com) on H100s and emits
artifacts that local `scripts/cuda_*` helpers consume. No local GPU needed.

### Prerequisites

1. `pip install -e ".[cuda-harness]"`
2. A Modal account configured via `modal setup` (confirm with
   `modal profile current`).
3. One-time weight cache on the Modal Volume:

   ```bash
   modal run -m cuda_harness.modal_common::download_inference_dependencies
   ```

### Harnesses

| Harness | What it does |
| --- | --- |
| `modal run -m cuda_harness.run_reference` | End-to-end CUDA inference for each (target, seed) pair; returns CIFs and score NPZ per diffusion sample. |
| `modal run -m cuda_harness.run_intermediates` | Same flow plus per-boundary tensor dumps (embedding, bond projection, per-recycle trunk, diffusion snapshots, confidence logits, ranking) into one NPZ per seed. |
| `modal run -m cuda_harness.run_determinism` | Runs chai-lab on CUDA twice on the same seed under configurable precision policy (`default` / `tf32_off` / `deterministic`) to measure CUDA's own run-to-run non-determinism. |
| `modal run -m cuda_harness.bench_throughput` | Per-module CUDA timings with warmup and `cuda.synchronize` gates; per-target JSON plus combined CSV. |

Local companions:

| Script | Question it answers |
| --- | --- |
| `scripts/cuda_parity.py` | Does MLX match CUDA tensor-for-tensor, stage by stage? |
| `scripts/cuda_structure_sweep.py` | How much do MLX and CUDA diverge on final 3D structures (RMSD / GDT / lDDT / ranking scores), optionally vs experimental PDB? |
| `scripts/cuda_error_accumulation.py` | How does MLX-vs-CUDA error grow across trunk recycles and diffusion steps? |
| `scripts/cuda_determinism_report.py` | How much of the MLX-vs-CUDA gap is just CUDA disagreeing with itself? |
| `scripts/cuda_mlx_diffusion_isolation.py` | How much of the structural gap is trunk drift vs MLX diffusion sampler differences? |
| `scripts/mlx_throughput.py` + `scripts/report_throughput_comparison.py` | Side-by-side MLX vs CUDA per-module wall clock. |
| `scripts/spawn_cuda_sweep.py` | Fan out every CUDA experiment to Modal in parallel. |

### Example workflows

```bash
# 1. Numerical parity (single target + seed, full stage-isolation diff)
modal run -m cuda_harness.run_intermediates --targets 1L2Y --seeds 42

python scripts/cuda_parity.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \
    --summary-json /tmp/chai_mlx_cuda/parity_1L2Y_seed42.json

# 2. Structure-level agreement across targets and seeds
modal run -m cuda_harness.run_reference \
    --targets 1L2Y,1VII,1CRN,1UBQ --seeds 0,42,123

python scripts/cuda_structure_sweep.py \
    --weights-dir weights \
    --reference-dir /tmp/chai_mlx_cuda/reference \
    --mlx-output-dir /tmp/chai_mlx_cuda/mlx \
    --compare-pdb --csv /tmp/chai_mlx_cuda/structure_sweep.csv

# 3. CUDA run-to-run determinism (answers "how much of the gap is just
#    CUDA disagreeing with itself?")
modal run -m cuda_harness.run_determinism \
    --targets 1L2Y --seeds 42 --precision default
python scripts/cuda_determinism_report.py \
    --npz /tmp/chai_mlx_cuda/determinism/1L2Y/seed_42_default.npz

# 4. Trunk-vs-diffusion attribution (answers "how much of the gap is
#    trunk drift vs MLX diffusion sampler differences?")
python scripts/cuda_mlx_diffusion_isolation.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \
    --cuda-reference-dir /tmp/chai_mlx_cuda/reference/1L2Y/seed_42

# 5. Throughput: CUDA on Modal, MLX locally, then a side-by-side report
modal run -m cuda_harness.bench_throughput \
    --targets 1L2Y,1VII,1CRN,1UBQ

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

All targets below are measured end-to-end in `HANDOFF.md` §1.1 – §1.5
(both ESM-on and ESM-off). Numbers here are Cα RMSD vs experimental PDB
with ESM on; see the HANDOFF for full ESM-off / ESM-on / MLX↔CUDA
matrices, the accumulation signature discussion, and the winners
breakdown (MLX 6 / CUDA 4 on the 10-target slate).

| Target        | Kind(s)              | Entities                                 | What it exercises                                        | MLX vs PDB (ESM on) |
| ------------- | -------------------- | ---------------------------------------- | -------------------------------------------------------- | ------------------- |
| `1L2Y`        | monomer              | 1 protein (20 aa)                        | Baseline Cα RMSD / GDT / lDDT                            | 0.74 Å              |
| `1VII`        | monomer              | 1 protein (35 aa)                        | Small α-helical fold                                     | 4.02 Å              |
| `1CRN`        | monomer              | 1 protein (46 aa)                        | Crambin, tight 2.0 Å PDB reference                       | 4.43 Å              |
| `1UBQ`        | monomer              | 1 protein (76 aa)                        | Mid-size α/β monomer                                     | 0.90 Å              |
| `1BRS`        | multimer             | 2 proteins (barnase + barstar, 199 aa)   | Multi-chain featurizer, interface Cα RMSD, iPTM          | 21.24 Å             |
| `1FKB`        | ligand               | FKBP-12 + FK506 SMILES                   | `EntityType.LIGAND` path, ligand heavy-atom RMSD         | 0.77 Å              |
| `7TIM`        | long                 | 1 protein (248 aa, TIM barrel)           | >200 residue ceiling, bf16 error growth                  | 6.49 Å              |
| `1BNA`        | dna, multimer        | 2 DNA strands (12 bp duplex)             | `EntityType.DNA` featurization, P-backbone RMSD          | 8.12 Å              |
| `1UBQ_ESM`    | esm                  | 1 protein (76 aa)                        | MLX esm2_t36_3B_UR50D vs chai-lab CUDA traced checkpoint | 0.90 Å              |
| `1CRN_CONSTR` | constraints, ligand  | 1 protein + methanethiol + 3 restraints  | Contact / pocket / covalent restraint features           | 4.72 Å              |

Axes that are plumbed but not yet end-to-end measured in this repo:
online MSA via the ColabFold API, online templates, offline MSA
directories, offline template m8 files, RNA-only inputs, and glycan
inputs. The `scripts/inference.py` CLI surface documents how to drive
each of them; see `HANDOFF.md` §1 and §8 for what that means for
confidence.

### Reproducing the expanded sweep

Everything below assumes you've installed the `[cuda-harness]` and
`[featurize]` extras and have a working `modal` profile.  No MSA or
template servers are used (offline only).  Local MLX inference uses the
`[esm]` extra when `--esm-backend mlx` is set.

```bash
# 1. CUDA reference runs for every new target on one seed (use_esm_embeddings=True
#    on the Modal side; local MLX stays esm-off unless you opt in below).
modal run -m cuda_harness.run_expanded_targets --seeds 42

# 2. Local MLX + per-kind structural comparison (reads the CUDA CIFs from step 1).
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
python scripts/inference.py --weights-dir weights/ \
    --fasta /tmp/chai_mlx_cuda/expanded/1UBQ_ESM/seed_42/input.fasta \
    --output-dir /tmp/chai_mlx_cuda/mlx_expanded/1UBQ_ESM_mlx \
    --esm-backend mlx
```

Step 2's CSV is wide: per-row it carries per-chain Cα RMSDs, an
interface Cα RMSD for multimers, a heavy-atom RMSD for ligands, and a
phosphate-backbone RMSD for DNA/RNA, alongside the aggregate MLX-vs-CUDA
score gap.  The per-kind summary printed at the end of the run is what
drops into this README once the sweep has been executed.

## License

Apache-2.0. This project is a derivative work of
[chai-lab](https://github.com/chaidiscovery/chai-lab) (also Apache-2.0);
see `NOTICE` for attribution.
