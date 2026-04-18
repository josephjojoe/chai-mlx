# chai-mlx

`chai-mlx` is an MLX port of [Chai-1](https://github.com/chaidiscovery/chai-lab),
the protein structure prediction model. It targets Apple silicon and ships
with a set of harnesses for comparing the MLX implementation against the
CUDA reference that actually runs in production.

## Status

- Structurally faithful end-to-end. Diffusion is bit-for-bit exact given
  correct trunk outputs.
- On 1L2Y (Trp-cage, 20 residues), MLX vs CUDA (H100, bf16) Cα RMSD is
  **0.75 Å mean** across 15 sample pairs (3 seeds × 5 diffusion samples),
  vs **0.57 Å** for CUDA against the NMR ground truth. MLX sits ~0.26 Å
  further from experimental truth than CUDA on average, with
  **GDT-TS = 95.1%** and **Cα lDDT = 89.8%** between the two implementations.
- The remaining gap is dominated by bf16 fused-kernel rounding accumulating
  through the 48-block pairformer trunk (inherent to running bf16 on two
  different implementations) plus per-step diffusion RNG differences between
  `mx.random.normal` and `torch.randn`. Neither is a structural bug.

Validated so far: monomers up to ~76 residues (1L2Y, 1VII, 1CRN, 1UBQ).
Multimer and ligand targets are untested.

## Install

```bash
git clone --recurse-submodules https://github.com/josephjojoe/chai-mlx
cd chai-mlx
pip install -e .
```

The `--recurse-submodules` flag fetches the pinned `chai-lab/` reference
checkout used by the featurizer and some comparison harnesses. If you've
already cloned without it, run `git submodule update --init chai-lab`.

Optional extras:

```bash
pip install -e ".[featurize]"     # torch + chai_lab; required for FASTA featurization
pip install -e ".[convert]"       # torch + safetensors; for TorchScript -> safetensors export
pip install -e ".[cuda-harness]"  # modal + gemmi + biopython; for the CUDA comparison harnesses
pip install -e ".[test]"          # pytest
```

## Quick start

```python
from chai_mlx import ChaiMLX, featurize_fasta

# Pulls ~1.2 GB of safetensors from the HF repo on first call.
model = ChaiMLX.from_pretrained("josephjojoe/chai-mlx")

ctx = featurize_fasta("input.fasta", output_dir="./out")  # needs [featurize] extra
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
  `python scripts/inference.py --weights-dir weights/ --fasta path/to/input.fasta`
- **Benchmark the diffusion loop**:
  `python examples/diffusion_benchmark.py`
- **FASTA featurization smoke**:
  `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- **Tests**: `pip install -e ".[test]"` then `pytest -q` (≈3 s; the
  `test_ranking.py` parity tests are auto-skipped unless `[featurize]`
  is also installed).

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

## License

Apache-2.0. This project is a derivative work of
[chai-lab](https://github.com/chaidiscovery/chai-lab) (also Apache-2.0);
see `NOTICE` for attribution.
