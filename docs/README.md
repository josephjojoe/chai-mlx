# Docs

- User-facing material lives in the root `README.md` and `examples/`.
- Contributor/reference material lives here in `docs/` plus `scripts/`.

| File | Purpose |
|------|---------|
| [`architecture.md`](./architecture.md) | Package map and module data flow |
| [`numerical_divergence.md`](./numerical_divergence.md) | Why Metal vs MPS matmul breaks the port, data, and proposals |
| [`matmul_tiling_divergence.md`](./matmul_tiling_divergence.md) | Deep dive: kernel source code, tiling orders, and what was tested |
| [`status.md`](./status.md) | Port status, isolation parity, memory, validation scripts |
| [`trunk_investigation_handoff_2026-04-14.md`](./trunk_investigation_handoff_2026-04-14.md) | Current trunk debugging handoff: context, prior fixes, harnesses, memory guidance, and remaining errors |
| [`weight_mapping.md`](./weight_mapping.md) | TorchScript → MLX naming and conversion checks |
| [`kernel_plan.md`](./kernel_plan.md) | Fast paths and kernel notes |

Reverse-engineering notes and graph dumps live under `findings/`.
