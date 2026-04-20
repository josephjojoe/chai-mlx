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

## Precision policy (read from these dumps, not from chai-lab docstrings)

Counting `aten::to(..., scalar_type_const)` calls in the IR:

| Module                   | bf16 casts | fp32 casts | Effective policy                          |
| ------------------------ | ---------: | ---------: | ----------------------------------------- |
| `trunk.pt`               |     18 324 |      5 160 | bf16 autocast w/ fp32 layer_norm/softmax  |
| `token_embedder.pt`      |       162* |          - | bf16 autocast                             |
| `confidence_head.pt`     |      1 102 |        355 | bf16 autocast w/ fp32 layer_norm/softmax  |
| `feature_embedding.pt`   |          0 |        297 | fp32                                      |
| `bond_loss_input_proj.pt`|          0 |          - | fp32 (trivial graph)                      |
| `diffusion_module.pt`    |          0 |         27 | **fp32 end-to-end** (only int/bool casts) |

*every ``aten::to(..., %const)`` in ``token_embedder`` resolves to scalar
type constant 15 (bf16).*

Reproduce the enumeration via ``awk 'BEGIN{b=0;f=0} /aten::to\(/{if (/%15[0-9]+/) b++; else f++} END{print b,f}' findings/graphs/<module>.txt`` (substituting the module's actual
bf16/fp32 constant variable names, which differ per graph — e.g. trunk
uses ``%15708``, ``%15695``). The companion probe
``cuda_harness/_probe_jit_precision.py`` resolves these at runtime via
TorchScript's C++ API for every exported bundle. The findings from the
drift investigation that motivated this section are in
[../drift_attribution.md](../drift_attribution.md).
