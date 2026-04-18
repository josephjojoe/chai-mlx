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
git clone --recurse-submodules <repo-url>
cd chai-mlx
pip install -e .
```

The `--recurse-submodules` flag fetches the pinned `chai-lab/` reference
checkout used by the featurizer and some comparison harnesses.

Optional extras:

```bash
pip install -e ".[featurize]"     # torch + chai_lab; required for FASTA featurization
pip install -e ".[convert]"       # torch + safetensors; for TorchScript -> safetensors export
pip install -e ".[cuda-harness]"  # modal + gemmi + biopython; for the CUDA comparison harnesses
pip install -e ".[test]"          # pytest
```

## Public API

```python
from chai_mlx import ChaiMLX, featurize, featurize_fasta

model = ChaiMLX.from_pretrained("./weights")

ctx = featurize(inputs)
# or:
# ctx = featurize_fasta("input.fasta", output_dir="./out")

result = model.run_inference(ctx, recycles=3, num_samples=5, num_steps=200)
# result.coords, result.confidence, result.ranking

# Debug variant returns the full context, embeddings, and trunk intermediates:
debug = model.run_inference_debug(ctx, recycles=3, num_samples=5, num_steps=200)
```

`ChaiMLX.from_pretrained(...)` accepts either a local directory containing
`config.json` plus `model.safetensors` (or sharded safetensors with an
index file), or a Hugging Face repo id, in which case `huggingface_hub`
downloads the snapshot first. The gitignored `weights/` directory is the
expected local home for model artifacts.

To build safetensors from the upstream Chai-1 TorchScript distribution:

```bash
chai-mlx-convert-torchscript --pt-dir path/to/pt --out-dir weights/
```

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
scripts/            contributor tooling (CUDA comparison drivers, inference)
tests/              pytest suite
chai-lab/           pinned upstream reference checkout (git submodule)
weights/            local model artifacts (gitignored)
```

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
| `modal run -m cuda_harness.run_reference` | End-to-end CUDA inference per target + seed; returns CIFs and score NPZ per diffusion sample. |
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

### Example workflow

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

# 3. Throughput: CUDA on Modal, MLX locally, then a side-by-side report
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

## Local workflows

- Smoke the package on random inputs: `python examples/basic_inference.py`
- Run end-to-end FASTA inference: `python scripts/inference.py --weights-dir weights/ --fasta path/to/input.fasta`
- Benchmark the diffusion loop: `python examples/diffusion_benchmark.py`
- FASTA featurization smoke: `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- Run the test suite: `pytest -q`

## License

Apache-2.0. This project is a derivative work of
[chai-lab](https://github.com/chaidiscovery/chai-lab) (also Apache-2.0);
see `NOTICE` for attribution.
