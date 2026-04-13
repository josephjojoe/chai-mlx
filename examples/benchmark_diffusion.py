from __future__ import annotations

import time

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize
from run_pipeline import build_dummy_inputs


def main() -> None:
    model = ChaiMLX()
    ctx = featurize(build_dummy_inputs())
    emb = model.embed_inputs(ctx)
    trunk = model.trunk(emb, recycles=1)
    cache = model.prepare_diffusion_cache(trunk)
    coords = model.init_noise(batch_size=1, num_samples=2, structure=emb.structure_inputs)
    schedule = list(model.schedule(num_steps=8))

    t0 = time.perf_counter()
    for sigma_curr, sigma_next, gamma in schedule:
        coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
        mx.eval(coords)
    t1 = time.perf_counter()
    print(f"{len(schedule)} diffusion steps: {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
