# Docs

- User-facing material lives in the root `README.md` and `examples/`.
- Contributor/reference material lives here in `docs/` plus `scripts/`.

| File | Audience | Purpose |
|------|----------|---------|
| [`../README.md`](../README.md) | users | install, public API, common workflows |
| [`architecture.md`](./architecture.md) | users + contributors | package map and module responsibilities |
| [`status.md`](./status.md) | contributors | port status, isolation parity numbers, path forward |
| [`weight_mapping.md`](./weight_mapping.md) | contributors | TorchScript → MLX naming and conversion checks |
| [`kernel_plan.md`](./kernel_plan.md) | contributors | performance notes, cache strategy, kernel tradeoffs |

Reverse-engineering notes and graph dumps live under `findings/`.
