# Weight mapping notes

See also [`status.md`](./status.md) and [`kernel_plan.md`](./kernel_plan.md).

This port tries to keep submodule names close to the upstream Chai-1 TorchScript hierarchy:

- `feature_embedding.*`
- `bond_projection.*`
- `token_input.*`
- `trunk_module.*`
- `diffusion_module.*`
- `confidence_head.*`

That is intentional so the conversion path can stay simple:

1. export each TorchScript module to NPZ,
2. inspect the exported keys,
3. either load directly if names match, or add a thin rename map,
4. validate per-component parity before stitching the full pipeline together.

## Important checks during conversion

- Linear weights should stay in `[out_dim, in_dim]` order.
- Explicitly verify LayerNorm parameter names and buffer names.
- Check any learned tensors that are not wrapped in `nn.Linear`, such as:
  - query biases,
  - output scalers,
  - Fourier embedding weights / biases.
- Confirm the confidence-head triangle attention layout separately because its fused projection differs from the trunk blocks.
