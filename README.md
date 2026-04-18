# Chai MLX

`chai-mlx` is a clean, installable MLX port of the [Chai-1](https://github.com/chaidiscovery/chai-lab)
protein structure prediction model. It is, as far as I know, the first
port of Chai-1 to Apple silicon, and it ships with a full set of
harnesses for verifying that the MLX path actually matches the CUDA
reference everyone runs in production.

## Current status

The short version ([`docs/status.md`](docs/status.md) has the full
story):

- Structurally faithful end-to-end. Diffusion is bit-for-bit exact when
  given correct trunk outputs.
- On 1L2Y (Trp-cage, 20 residues), median Cα spacing is within **0.10 Å**
  of the Torch-MPS reference across three seeds in both bf16 and fp32.
- The remaining MPS gap is traced to **different fused-kernel rounding
  in bf16 `sigmoid` / `silu`** between MLX and MPS — not an algorithmic
  difference (all ops are bit-identical in fp32; `exp` is bit-identical
  even in bf16).
- **The MPS comparison is not what ships to users.** Nobody runs Chai-1
  on MPS in production; everyone uses CUDA. Until recently we had no
  data on how MLX compares to CUDA, and running the original TorchScript
  stack end-to-end on this 16 GB MacBook now OOMs (it is very memory
  unoptimised). So:
- **A full MLX-vs-CUDA comparison harness runs on [Modal](https://modal.com)
  instead.** See `cuda_harness/` below. Modal is the serverless
  platform we use to drive H100 reference runs; we don't need any local
  GPU access to verify MLX against CUDA.

## Quick start

```bash
git clone <repo-url>
cd chai-mlx
pip install -e .
```

Optional extras:

```bash
pip install -e ".[featurize]"   # torch + chai_lab, required for FASTA featurization
pip install -e ".[convert]"     # torch + safetensors, for TorchScript -> safetensors export
pip install -e ".[cuda-harness]" # modal + gemmi + biopython, for the CUDA comparison harnesses
```

## Public API

```python
from chai_mlx import ChaiMLX, featurize, featurize_fasta

model = ChaiMLX.from_pretrained("./weights")

ctx = featurize(inputs)
# or:
# ctx = featurize_fasta("input.fasta", output_dir="./out")

# Production inference API (does not retain intermediates):
result = model.run_inference(ctx, recycles=3, num_samples=1, num_steps=200)

# Debug API (returns full context/embedding/trunk intermediates):
debug_result = model.run_inference_debug(ctx, recycles=3, num_samples=1, num_steps=200)
```

## Repository layout

```text
chai_mlx/
  config.py              # shared architecture/config dataclasses
  utils.py               # numerics, masking, geometry, schedule helpers
  data/                  # featurization adapters and typed contexts
  model/                 # public model pipeline and major model stages
  nn/                    # reusable neural-network building blocks and kernels
  io/weights/            # weight export, conversion, loading, validation

examples/                # minimal runnable examples (basic inference, diffusion bench, FASTA smoke)
scripts/                 # contributor tooling: parity harnesses, weight conversion, CUDA comparison drivers
cuda_harness/            # Modal-hosted chai-lab-on-CUDA reference harness (optional)
docs/                    # architecture, weight mapping, status, kernel plan
weights/                 # local model artifacts (gitignored)
findings/                # deeper architecture notes + extracted TorchScript graphs
chai-lab/                # (gitignored) local Chai Discovery reference checkout used by some harnesses
```

## Weights

`ChaiMLX.from_pretrained(...)` accepts either:

- a local directory containing `config.json` plus `model.safetensors` or
  sharded safetensors, or
- a Hugging Face repo id, in which case `huggingface_hub` is used to
  download the snapshot first.

The repo-level `weights/` directory is the expected local home for model
artifacts during development and is gitignored. If you are starting from
the TorchScript distribution, use
`chai-mlx-convert-torchscript --pt-dir path/to/pt --out-dir weights/`
to build safetensors that `from_pretrained` understands.

## CUDA comparison harnesses (Modal)

This is the big new piece of infrastructure in the repo. Everything in
`cuda_harness/` runs on Modal's serverless platform and produces
outputs that the local `scripts/cuda_*` helpers consume. Nothing here
needs a local GPU.

Prerequisites:

1. `pip install -e ".[cuda-harness]"`
2. Have a Modal account and profile set up (`modal setup`). Confirm
   with `modal profile current`.
3. Once per workspace, prime the weight cache on the Modal Volume:

   ```bash
   modal run -m cuda_harness.modal_common::download_inference_dependencies
   ```

   This downloads the ~7 GB of chai-1 TorchScript checkpoints from
   Chai Discovery's CDN into the `chai-mlx-weights` Modal Volume so
   subsequent inference runs don't re-pay the cost.

Harness | What it does | Entry point
------- | ------------ | -----------
Reference inference | Runs `chai_lab.run_inference` on CUDA for any number of targets/seeds and returns the CIFs + score NPZ per diffusion sample. | `modal run -m cuda_harness.run_reference`
Intermediates capture | Same flow but also dumps every per-module boundary tensor (feature embedding, bond projection, token embedder, each trunk recycle, diffusion schedule + per-step snapshots, confidence logits, per-sample ranking) into a single NPZ per seed. | `modal run -m cuda_harness.run_intermediates`
Per-module throughput | Times every stage with warmup and `cuda.synchronize` gates. Produces a per-target JSON plus a combined CSV. | `modal run -m cuda_harness.bench_throughput`

The local-side companions each target a specific research question.
They all expect one of the Modal outputs above.

Question | Harness
-------- | -------
Does the MLX port match CUDA tensor-for-tensor, stage by stage? | [`scripts/cuda_parity.py`](scripts/cuda_parity.py)
How much do MLX and CUDA diverge on final 3D structures (RMSD/GDT/lDDT/ranking scores), optionally vs experimental PDB? | [`scripts/cuda_structure_sweep.py`](scripts/cuda_structure_sweep.py)
How does MLX-vs-CUDA error grow across trunk recycles and diffusion steps? Is MLX's denoise step of similar magnitude to CUDA's? | [`scripts/cuda_error_accumulation.py`](scripts/cuda_error_accumulation.py)
How fast is each MLX stage locally, for a directly comparable side-by-side table with CUDA? | [`scripts/mlx_throughput.py`](scripts/mlx_throughput.py) + [`scripts/report_throughput_comparison.py`](scripts/report_throughput_comparison.py)

Common workflows:

```bash
# 1. Numerical parity on a single target+seed (feeds CUDA intermediates into MLX at every boundary)
modal run -m cuda_harness.run_intermediates --targets 1L2Y --seeds 42 \
    --snapshot-steps first,mid,last \
    --output-dir /tmp/chai_mlx_cuda/intermediates

python scripts/cuda_parity.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \
    --summary-json /tmp/chai_mlx_cuda/parity_1L2Y_seed42.json

# 2. Structure-level agreement across targets and seeds
modal run -m cuda_harness.run_reference \
    --targets 1L2Y,1VII,1CRN,1UBQ --seeds 0,42,123 \
    --output-dir /tmp/chai_mlx_cuda/reference

python scripts/cuda_structure_sweep.py \
    --weights-dir weights \
    --reference-dir /tmp/chai_mlx_cuda/reference \
    --mlx-output-dir /tmp/chai_mlx_cuda/mlx \
    --compare-pdb --csv /tmp/chai_mlx_cuda/structure_sweep.csv

# 3. Error accumulation along the trunk-recycle and diffusion-step axes
modal run -m cuda_harness.run_intermediates --targets 1L2Y --seeds 42 \
    --snapshot-steps 1,25,50,100,150,199,200

python scripts/cuda_error_accumulation.py \
    --weights-dir weights \
    --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \
    --mode cascading --csv /tmp/chai_mlx_cuda/error_accumulation.csv

# 4. Throughput: CUDA on Modal, MLX locally, then a side-by-side report
modal run -m cuda_harness.bench_throughput \
    --targets 1L2Y,1VII,1CRN,1UBQ --output-dir /tmp/chai_mlx_cuda/throughput

python scripts/mlx_throughput.py \
    --weights-dir weights \
    --targets 1L2Y,1VII,1CRN,1UBQ \
    --output-json /tmp/chai_mlx_cuda/throughput/mlx.json

python scripts/report_throughput_comparison.py \
    --mlx-json /tmp/chai_mlx_cuda/throughput/mlx.json \
    --cuda-dir /tmp/chai_mlx_cuda/throughput
```

## Local-only workflows

- Smoke test the package: `python examples/basic_inference.py`
- Run end-to-end FASTA inference: `python scripts/inference.py --weights-dir weights/ --fasta path/to/input.fasta`
- Benchmark the diffusion loop: `python examples/diffusion_benchmark.py`
- Run the FASTA path: `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- Run TorchScript/MLX parity checks (vs local TorchScript): `python scripts/parity_check.py --torchscript-dir ... --safetensors-dir ...`
- Generate FASTA-backed reference inputs/tensors from `chai-lab`: `python scripts/chai_lab_reference_dump.py --input-npz ... --reference-npz ...`
- Run dump-based layer parity checks: `python scripts/layer_parity.py --weights-dir weights/ --input-npz ... --reference-npz ...`
- Bisect one denoise call and run the zero-input diagnostic: `python scripts/bisect_denoise.py --weights-dir weights/ --input-npz ... --reference-npz ...`
- Run the deep denoise checkpoint trace: `python scripts/deep_denoise_trace.py --weights-dir weights/ --input-npz ... --reference-npz ...`
- Run diffusion-specific diagnostics and full-loop spacing checks: `python scripts/diffusion_diagnostics.py --weights-dir weights/ --experiment perop|lyapunov|hybrid|all`
- Run convert/load/smoke validation on real TorchScript artifacts: `python scripts/weight_loading_e2e.py --torchscript-dir ...`
- Run the targeted denoise dataflow regression test: `pytest -q tests/test_diffusion.py`
- Export TorchScript weights to NPZ: `chai-mlx-export-torchscript src.pt out.npz`
- Convert `.pt` shards directly to safetensors: `chai-mlx-convert-torchscript --pt-dir path/to/pt --out-dir weights/`
- Convert NPZ weights to safetensors: `chai-mlx-convert-npz npz_dir weights/`

## Notes

- The MLX neural core is separate from the upstream bioinformatics
  frontend. `featurize_fasta(...)` delegates to `chai_lab` rather than
  reimplementing that stack.
- `examples/` stays user-facing while `scripts/` holds contributor
  tooling such as parity and weight-conversion helpers.
- `bisect_denoise.py` and `layer_parity.py` are the closest mirrors of
  the actual MLX runtime path for denoise debugging.
- `deep_denoise_trace.py` is a manual replay harness for checkpoint
  localization. If it disagrees with direct top-level wrapper parity,
  treat the wrapper parity result as authoritative and fix the trace
  harness before drawing conclusions.
- The authoritative status page is [`docs/status.md`](docs/status.md).
- The authoritative harness map is this README; the CUDA harness
  entries above are kept in sync with `cuda_harness/` and
  `scripts/cuda_*`.
