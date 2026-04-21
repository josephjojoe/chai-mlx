# Graph Dumps

TorchScript graph dumps extracted from the Chai-1 model. Each of the six major
modules has four companion files:

| Suffix | Format | Contents |
|---|---|---|
| `.txt` | TorchScript IR | Raw intermediate-representation graph (`prim::Constant`, `aten::*` ops, etc.) |
| `_code.txt` | Python | Decompiled `forward` methods for **every submodule** in the module hierarchy |
| `_toplevel_code.txt` | Python | Decompiled top-level `forward_256` and its immediate helper calls only |
| `_forward256.py` | Python | Fully inlined `forward_256` as a single executable function (all weights pre-loaded, no sub-calls) |

## Modules

| Module | Role |
|---|---|
| `trunk` | Pairformer stack (48 blocks of single/pair attention + transitions) |
| `token_embedder` | Embeds raw input features into token-level representations |
| `feature_embedding` | Produces initial single/pair representations from input features |
| `diffusion_module` | Denoising diffusion over atom coordinates |
| `confidence_head` | Predicts pLDDT, PAE, and other confidence metrics |
| `bond_loss_input_proj` | Projects inputs for the bond-geometry loss |

## Precision policy

These dumps are the raw source material for inspecting the reference
bundle's precision boundary. Historical cast-count summaries have been
removed so the repo does not carry stale precision results after recent
changes.

To regenerate the current summary:

1. Inspect the `aten::to(...)` sites in the IR under this directory.
2. Run `cuda_harness/_probe_jit_precision.py` to resolve the current
   scalar-type constants programmatically.
