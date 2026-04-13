# Chai MLX

`chai-mlx` is a cleaned-up MLX port of the Chai-1 inference stack. The repo is now installable from the clone root and organized so the public entry points, model code, data adapters, and developer tooling are easier to navigate.

## Quick start

```bash
git clone <repo-url>
cd chai-mlx
pip install -e .
```

Optional extras:

```bash
pip install -e ".[featurize]"
pip install -e ".[convert]"
```

- `.[featurize]` adds `torch` and `chai_lab` for FASTA-driven featurization.
- `.[convert]` adds the conversion stack for exporting TorchScript weights and writing safetensors.

## Public API

```python
from chai_mlx import ChaiMLX, featurize, featurize_fasta

model = ChaiMLX.from_pretrained("./weights")

ctx = featurize(inputs)
# or:
# ctx = featurize_fasta("input.fasta", output_dir="./out")

emb = model.embed_inputs(ctx)
trunk = model.trunk(emb, recycles=3)
cache = model.prepare_diffusion_cache(trunk)

coords = model.init_noise(
    batch_size=emb.token_single_input.shape[0],
    num_samples=1,
    structure=emb.structure_inputs,
)
for sigma_curr, sigma_next, gamma in model.schedule(num_steps=200):
    coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)

confidence = model.confidence(trunk, coords)
ranking = model.rank_outputs(confidence, coords, emb.structure_inputs)
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
examples/
  basic_inference.py     # minimal end-to-end dummy-input smoke example
  diffusion_benchmark.py # diffusion loop timing harness
  fasta_smoke.py        # FASTA -> feature embedding smoke script
scripts/
  parity_check.py       # per-component TorchScript vs MLX parity harness
  chai_lab_reference_dump.py # FASTA-backed reference dump generator from chai-lab
  layer_parity.py       # dump-based intermediate tensor parity harness
  weight_loading_e2e.py # convert/load/smoke validation against real weights
docs/
  README.md              # docs map
  architecture.md        # package map and responsibilities
  status.md              # current parity / production-readiness notes
weights/                 # local model artifacts (gitignored)
findings/                # deeper architecture and optimization notes
```

## Weights

`ChaiMLX.from_pretrained(...)` accepts either:

- a local directory containing `config.json` plus `model.safetensors` or sharded safetensors, or
- a Hugging Face repo id, in which case `huggingface_hub` is used to download the snapshot first.

The repo-level `weights/` directory is intended for local model artifacts during development and is ignored by git.

## Common workflows

- Smoke test the package: `python examples/basic_inference.py`
- Benchmark the diffusion loop: `python examples/diffusion_benchmark.py`
- Run the FASTA path: `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- Run TorchScript/MLX parity checks: `python scripts/parity_check.py --torchscript-dir ... --safetensors-dir ...`
- Generate FASTA-backed reference inputs/tensors from `chai-lab`: `python scripts/chai_lab_reference_dump.py --input-npz ... --reference-npz ...`
- Run dump-based layer parity checks: `python scripts/layer_parity.py --weights-dir weights/ --input-npz ... --reference-npz ...`
- Run convert/load/smoke validation on real TorchScript artifacts: `python scripts/weight_loading_e2e.py --torchscript-dir ...`
- Export TorchScript weights to NPZ: `chai-mlx-export-torchscript src.pt out.npz`
- Convert `.pt` shards directly to safetensors: `chai-mlx-convert-torchscript --pt-dir path/to/pt --out-dir weights/`
- Convert NPZ weights to safetensors: `chai-mlx-convert-npz npz_dir weights/`

## Notes

- The MLX neural core is separate from the upstream bioinformatics frontend. `featurize_fasta(...)` delegates to `chai_lab` rather than reimplementing that stack.
- The repo is organized so `examples/` stays user-facing while `scripts/` holds contributor tooling such as parity and weight-conversion helpers.
- The authoritative status page is [`docs/status.md`](docs/status.md).
