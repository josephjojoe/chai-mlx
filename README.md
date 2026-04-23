# chai-mlx

`chai-mlx` is an [MLX](https://github.com/ml-explore/mlx) inference port of
[Chai-1](https://github.com/chaidiscovery/chai-lab) for Apple silicon. It keeps
chai-lab's FASTA/MSA/template/constraint featurization path, runs the model in
MLX, and supports proteins, ligands, DNA, RNA, glycans, and optional ESM-2
embeddings.

## Status

- End-to-end MLX inference is available through `chai-mlx-infer`.
- Optional ESM cache generation is available through `chai-mlx-precompute-esm`.
- Public precision policies are `reference` and `float32`.

## Requirements

- Python >= 3.11
- [MLX](https://github.com/ml-explore/mlx) >= 0.16
- Apple silicon recommended
- About 1.2 GB of disk for model weights

The base install already includes `torch` and the pinned `chai_lab` dependency
needed for `featurize_fasta()` and `chai-mlx-infer`.

## Install

Clone-first is the intended workflow:

```bash
git clone https://github.com/josephjojoe/chai-mlx
cd chai-mlx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

That editable install is also the minimum contributor setup: it pulls in the
`torch` + pinned `chai_lab` runtime that powers `featurize_fasta()`,
`chai-mlx-infer`, and the runtime-dependent pytest cases. If you run `pytest`
from a bare checkout (or after a `--no-deps` install), those tests will be
reported as skipped because the chai-lab runtime is not importable.

For the default contributor test environment, install the `test` extra:

```bash
python -m pip install -e ".[test]"
pytest -q
```

If you also want MLX-native ESM-2 embeddings or the precompute cache workflow:

```bash
python -m pip install -e ".[esm]"
```

If you want the optional ESM adapter tests as well, install both extras:

```bash
python -m pip install -e ".[test,esm]"
pytest -q
```

No git submodules are required. The base install resolves the pinned
`chai_lab` dependency directly, and the `esm` extra resolves `esm_mlx`
directly.

## Quick Start

### Python API

```python
from chai_mlx import ChaiMLX, featurize_fasta

model = ChaiMLX.from_pretrained("josephjojoe/chai-mlx")
ctx = featurize_fasta("input.fasta", output_dir="./out")
result = model.run_inference(ctx, recycles=3, num_samples=5, num_steps=200)
```

`ChaiMLX.from_pretrained(...)` accepts either a Hugging Face repo id or a local
directory containing `config.json` plus `model.safetensors` (or sharded
safetensors with an index file).

### Command line

```bash
chai-mlx-infer \
    --weights-dir josephjojoe/chai-mlx \
    --fasta path/to/input.fasta \
    --output-dir ./out
```

This writes:

```text
out/
  input.fasta
  pred.model_idx_{0..N}.cif
  scores.json
  scores.model_idx_{0..N}.npz
  manifest.json
  _features/
```

Useful flags:

- `--num-samples 5`
- `--num-trunk-samples 1`
- `--recycles 3 --num-steps 200`
- `--dtype reference|float32`
- `--pad-strategy exact|bucket`
- `--constraint-path restraints.csv`
- `--use-msa-server` or `--msa-directory <dir>`
- `--templates-path <file>` or `--use-templates-server`
- `--esm-backend {off,chai,mlx,mlx_cache}`
- `--esm-cache-dir <dir>`
- `--fasta-dir <dir>`
- `--write-msa-plot`

Run `chai-mlx-infer --help` for the full surface.

## Pad Strategy

Chai-1's traced bundle uses seven fixed token sizes:
`256, 384, 512, 768, 1024, 1536, 2048`, with the atom axis set to
`23 * n_tokens`.

`chai-mlx-infer` defaults to `--pad-strategy exact`, which keeps `n_tokens`
equal to the real token count and pads `n_atoms` only to the next multiple of
32. This is the fastest option for normal MLX inference.

Use `--pad-strategy bucket` if you want those same seven traced bucket sizes.

## FASTA Format

Chai-lab's `>kind|name=SHORT` header grammar is enforced up front.

```text
>protein|name=A
MKFLILFNILVSTLSFSSAQA
>ligand|name=LIG
CC(=O)Oc1ccccc1C(=O)O
>dna|name=D1
ACGTACGTACGT
```

`kind` must be one of `protein`, `ligand`, `dna`, `rna`, or `glycan`.

## ESM-2 Embeddings

`featurize_fasta()` and `chai-mlx-infer` expose four ESM modes:

- `off`: zero-fill the ESM feature.
- `chai`: use chai-lab's traced CUDA checkpoint.
- `mlx`: run `esm-mlx` in-process.
- `mlx_cache`: load pre-computed `<sha1(seq)>.npy` embeddings.

The cache workflow is:

```bash
chai-mlx-precompute-esm \
    --fasta path/to/input.fasta \
    --cache-dir ./esm_cache

chai-mlx-infer \
    --weights-dir josephjojoe/chai-mlx \
    --fasta path/to/input.fasta \
    --output-dir ./out \
    --esm-backend mlx_cache \
    --esm-cache-dir ./esm_cache
```

`mlx_cache` is the most practical option on 16 GB Macs because the inference
process does not need to keep ESM-2 3B weights in memory.

## Weights

Pre-converted weights are hosted on Hugging Face at
[`josephjojoe/chai-mlx`](https://huggingface.co/josephjojoe/chai-mlx).
`ChaiMLX.from_pretrained(...)` downloads them on first use and reuses the
standard Hugging Face cache afterwards.

## Example And Tests

- FASTA smoke: `python examples/fasta_smoke.py --fasta path/to/input.fasta`
- Default test suite: `python -m pip install -e ".[test]" && pytest -q`
- ESM adapter coverage too: `python -m pip install -e ".[test,esm]" && pytest -q`
- Slow end-to-end inference: `CHAI_MLX_RUN_SLOW=1 pytest -q -m slow`

## License

Apache-2.0. This project is a derivative work of
[chai-lab](https://github.com/chaidiscovery/chai-lab); see `NOTICE` for
attribution.
