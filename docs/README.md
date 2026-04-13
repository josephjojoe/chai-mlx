# Docs Map

Use this split:

- User-facing material lives in `README.md` and the workflow scripts under `examples/`.
- Contributor/reference material lives here in `docs/` plus the heavier tooling under `scripts/`.

| File | Audience | Purpose |
|------|----------|---------|
| [`../README.md`](../README.md) | users | install, public API, common workflows |
| [`architecture.md`](./architecture.md) | users + contributors | package map and module responsibilities |
| [`status.md`](./status.md) | contributors | parity notes, validation status, production checklist |
| [`weight_mapping.md`](./weight_mapping.md) | contributors | TorchScript -> MLX naming and conversion checks |
| [`kernel_plan.md`](./kernel_plan.md) | contributors | performance notes, cache strategy, kernel tradeoffs |
| [`featurization_audit.md`](./featurization_audit.md) | contributors | historical audit trail for the featurization path |

Deeper reverse-engineering notes and graph dumps still live under `findings/`.
