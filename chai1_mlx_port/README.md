# chai1-mlx

A logically structured MLX inference port for Chai-1 with:

- a clean pipeline API,
- explicit diffusion-time caches,
- MLX-native `nn.Module` blocks,
- optional custom Metal kernels for the nonstandard hot paths,
- a weight-export path from the upstream TorchScript artifacts.

## Design choices

This port intentionally separates **frontend featurization** from the **MLX neural core**:

- `featurize(...)` accepts either a precomputed `FeatureContext` or a dict of already-encoded tensors.
- `featurize_fasta(...)` delegates to **chai-lab's** featurization pipeline (parsing, tokenization, MSA/template/ESM loading, feature generation, collation) and converts the result into a `FeatureContext`.  This avoids reimplementing the 30+ feature generators and guarantees correctness.  Requires `torch` and `chai_lab` at runtime (`pip install .[featurize]`).
- The neural core starts from the **final encoded feature tensors** (TOKEN/TOKEN_PAIR/ATOM/ATOM_PAIR/MSA/TEMPLATES + bond adjacency).
- This keeps the MLX implementation focused on the model itself rather than the bioinformatics pipeline.

That means the API stays close to:

```python
# From FASTA (uses chai-lab under the hood):
ctx = featurize_fasta("input.fasta", output_dir="./out")

# Or from precomputed tensors:
ctx = featurize(inputs)

emb = model.embed_inputs(ctx)
trunk_out = model.trunk(emb, recycles=3)

diff_cache = model.prepare_diffusion_cache(trunk_out, emb.structure_inputs)

coords = model.init_noise(batch_size=1, num_samples=5, structure=emb.structure_inputs)
for sigma_curr, sigma_next, gamma in model.schedule(num_steps=200):
    coords = model.diffusion_step(
        cache=diff_cache,
        coords=coords,
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
        gamma=gamma,
    )

conf = model.confidence(trunk_out, coords)
rank = model.rank_outputs(conf, coords, emb.structure_inputs)
```

## Layout

```text
src/chai1_mlx/
  api.py                # high-level pipeline wrapper
  config.py             # dimensions and architecture constants
  types.py              # typed dataclasses for contexts and outputs
  featurize.py          # frontend adapters: precomputed tensors + chai-lab FASTA path
  embeddings.py         # feature embedding + bond projection + token embedder
  trunk.py              # template embedder, MSA module, pairformer stack
  diffusion.py          # diffusion conditioning, cache split, sampler step
  confidence.py         # confidence head
  ranking.py            # pTM / ipTM / clash-based ranking
  utils.py              # masking, reshaping, binning, segment reductions
  layers/
    common.py           # LayerNorm/AdaLN/SwiGLU/gating helpers
    attention.py        # pair-biased token attention and diffusion attention
    triangle.py         # triangle multiplication + triangle attention
    atom_attention.py   # blocked atom attention and atom encoder/decoder
    pairformer.py       # reusable pairformer block
  kernels/
    sources.py          # Metal source snippets
    elementwise.py      # fused SwiGLU / gate+residual / AdaLN wrappers
  weights/
    export_torchscript.py
                        # TorchScript -> NPZ export
    convert_to_safetensors.py
                        # NPZ -> safetensors conversion
    name_map.py         # TorchScript -> MLX parameter name mapping
    load.py             # component loading helpers
    validate.py         # weight shape validation
  examples/
    run_pipeline.py     # end-to-end example with precomputed tensors
    benchmark_diffusion.py
                        # cache + diffusion loop benchmark harness
    validate_parity.py  # per-component numerical parity vs TorchScript
```

## Current status

This repository is a **serious implementation scaffold** for an MLX port, not a drop-in tested replacement for the upstream CUDA runtime.

What is already here:

- all major Chai-1 model stages are represented as MLX modules;
- the diffusion path is explicitly split into
  `prepare_diffusion_cache(...)` and `diffusion_step(...)`;
- pair conditioning, per-block pair biases, and blocked atom-pair bases are cached;
- the main pair-biased attentions use `mlx.core.fast.scaled_dot_product_attention`;
- custom Metal kernels are supplied for easy fusion wins and an experimental blocked local attention path.

What you still need for a production run:

- export the upstream TorchScript weights with `weights/export_torchscript.py`,
- validate the exact name mapping against your local Chai-1 artifact versions,
- run numerical parity tests against the PyTorch reference,
- benchmark the experimental Metal kernels on your target Apple silicon GPU.

## Practical recommendation

For best time-to-first-result on Apple silicon:

1. keep **global attention** on MLX's fused SDPA path,
2. cache all diffusion-time pair conditioning,
3. use the custom Metal kernels for elementwise fusion and blocked atom attention,
4. introduce chunked triangle multiplication once memory becomes the bottleneck.
