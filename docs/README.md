# Docs

- User-facing material lives in the root `README.md` and `examples/`.
- Contributor/reference material lives here in `docs/` plus `scripts/`.

| File | Purpose |
|------|---------|
| [`status.md`](./status.md) | Canonical parity status: what's been fixed, current error budget, diagnosis of the remaining bf16 fused-kernel gap, harnesses, and next steps |
| [`architecture.md`](./architecture.md) | Package map and module data flow |
| [`weight_mapping.md`](./weight_mapping.md) | TorchScript → MLX naming and conversion checks |
| [`kernel_plan.md`](./kernel_plan.md) | Fast paths and kernel notes |

Reverse-engineering notes and graph dumps live under `findings/`.

## Historical note

Earlier docs (`numerical_divergence.md`, `matmul_tiling_divergence.md`, and
the pre-April-14 `status.md`) attributed the MLX-vs-MPS gap to matmul tiling
order and claimed the port could not produce valid protein structures. Both
claims were falsified by the April 14 fixes and the exhaustive bf16 kernel
enumeration documented in `status.md` (`matmul` and `exp` are bit-identical
across backends; only fused `sigmoid`/`silu`/`softmax` differ in bf16). Those
three files have been deleted; their content is recoverable from git history
if needed (`git log --diff-filter=D --follow -- docs/numerical_divergence.md`).
