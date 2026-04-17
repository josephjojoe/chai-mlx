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
