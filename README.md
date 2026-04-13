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
  run_pipeline.py        # minimal end-to-end dummy-input smoke example
  benchmark_diffusion.py # diffusion loop timing harness
  featurize_fasta.py     # FASTA -> feature embedding smoke script
  validate_parity.py     # TorchScript vs MLX parity harness
scripts/
  convert_weights.py     # memory-aware TorchScript -> safetensors conversion
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

- Smoke test the package: `python examples/run_pipeline.py`
- Benchmark the diffusion loop: `python examples/benchmark_diffusion.py`
- Run the FASTA path: `python examples/featurize_fasta.py --fasta path/to/input.fasta`
- Export TorchScript weights to NPZ: `chai-mlx-export-torchscript src.pt out.npz`
- Convert NPZ weights to safetensors: `chai-mlx-convert-npz npz_dir weights/`
- Convert `.pt` shards directly with the memory-aware helper: `python scripts/convert_weights.py --pt-dir path/to/pt --out-dir weights`

## Notes

- The MLX neural core is separate from the upstream bioinformatics frontend. `featurize_fasta(...)` delegates to `chai_lab` rather than reimplementing that stack.
- The repo is much closer to a usable developer package now, but numerical parity and production validation still live in the docs and examples rather than a full automated test suite.
- The authoritative status page is [`docs/status.md`](docs/status.md).
