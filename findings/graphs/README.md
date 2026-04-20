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
