from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Sequence

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from chai_mlx.config import ChaiConfig


_DTYPE_MAP = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


def resolve_dtype(cfg_or_str: "ChaiConfig | str") -> mx.Dtype:
    """Resolve a config or string dtype name to an ``mx.Dtype``."""
    name = cfg_or_str if isinstance(cfg_or_str, str) else cfg_or_str.compute_dtype
    try:
        return _DTYPE_MAP[name]
    except KeyError:
        raise ValueError(f"Unknown compute_dtype {name!r}; expected one of {list(_DTYPE_MAP)}")


def ensure_fp32(x: mx.array) -> mx.array:
    return x if x.dtype == mx.float32 else x.astype(mx.float32)


def chunk_last(x: mx.array, chunks: int) -> list[mx.array]:
    size = x.shape[-1]
    assert size % chunks == 0, (size, chunks)
    step = size // chunks
    return [x[..., i * step : (i + 1) * step] for i in range(chunks)]


def split_heads(x: mx.array, num_heads: int, head_dim: int) -> mx.array:
    return x.reshape(*x.shape[:-1], num_heads, head_dim)


def merge_heads(x: mx.array) -> mx.array:
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


def masked_mean(
    x: mx.array,
    mask: mx.array,
    axis: int | Sequence[int],
    *,
    keepdims: bool = False,
    eps: float = 1e-4,
) -> mx.array:
    weights = mask.astype(x.dtype)
    while weights.ndim < x.ndim:
        weights = mx.expand_dims(weights, axis=-1)
    num = mx.sum(x * weights, axis=axis, keepdims=keepdims)
    den = mx.sum(weights, axis=axis, keepdims=keepdims)
    den = mx.maximum(den, mx.array(eps, dtype=x.dtype))
    return num / den


def make_additive_mask(
    mask: mx.array,
    masked_value: float = -10000.0,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    bool_mask = mask.astype(mx.bool_) if mask.dtype != mx.bool_ else mask
    dt = dtype if dtype is not None else mx.float32
    return mx.where(
        bool_mask,
        mx.zeros(mask.shape, dtype=dt),
        mx.full(mask.shape, masked_value, dtype=dt),
    )


def stable_softmax(logits: mx.array, axis: int = -1) -> mx.array:
    logits = ensure_fp32(logits)
    logits = logits - mx.max(logits, axis=axis, keepdims=True)
    exps = mx.exp(logits)
    return exps / mx.sum(exps, axis=axis, keepdims=True)


def pairwise_distance(x: mx.array, y: mx.array | None = None, eps: float = 1e-10) -> mx.array:
    y = x if y is None else y
    diff = x[..., :, None, :] - y[..., None, :, :]
    return mx.sqrt(mx.maximum(mx.sum(diff * diff, axis=-1), eps))


def cdist(x: mx.array, y: mx.array | None = None, eps: float = 1e-10) -> mx.array:
    return pairwise_distance(x, y, eps=eps)


def gather_tokens_to_atoms(token_values: mx.array, atom_token_index: mx.array) -> mx.array:
    batch = mx.arange(token_values.shape[0])[:, None]
    return token_values[batch, atom_token_index]


def gather_blocked_atom_values(atom_values: mx.array, indices: mx.array) -> mx.array:
    batch = mx.arange(atom_values.shape[0])[:, None, None]
    return atom_values[batch, indices]


def gather_blocked_pair_values(token_pair_values: mx.array, q_idx: mx.array, kv_idx: mx.array) -> mx.array:
    batch = mx.arange(token_pair_values.shape[0])[:, None, None, None]
    q = q_idx[:, :, :, None]
    kv = kv_idx[:, :, None, :]
    return token_pair_values[batch, q, kv]


def segment_sum(
    values: mx.array,
    segment_ids: mx.array,
    num_segments: int,
    mask: mx.array | None = None,
) -> mx.array:
    weights = mx.eye(num_segments, dtype=values.dtype)[segment_ids]
    if mask is not None:
        w = mask.astype(values.dtype)
        while w.ndim < weights.ndim:
            w = mx.expand_dims(w, axis=-1)
        weights = weights * w
    return mx.einsum("ban,bad->bnd", weights, values)


def segment_mean(
    values: mx.array,
    segment_ids: mx.array,
    num_segments: int,
    mask: mx.array | None = None,
    eps: float = 1.0,
) -> mx.array:
    sums = segment_sum(values, segment_ids, num_segments, mask=mask)
    ones = mx.ones((*values.shape[:-1], 1), dtype=values.dtype)
    counts = segment_sum(ones, segment_ids, num_segments, mask=mask)
    counts = mx.maximum(counts, mx.array(eps, dtype=values.dtype))
    return sums / counts


def one_hot_binned(values: mx.array, edges: Iterable[float], *, dtype: mx.Dtype = mx.float32) -> mx.array:
    edges_arr = mx.array(list(edges), dtype=dtype)
    idx = mx.sum(values[..., None] > edges_arr, axis=-1).astype(mx.int32)
    return mx.eye(edges_arr.shape[0] + 1, dtype=dtype)[idx]


def representative_atom_coords(coords: mx.array, token_reference_atom_index: mx.array) -> mx.array:
    batch = mx.arange(coords.shape[0])[:, None]
    return coords[batch, token_reference_atom_index]


def expand_plddt_to_atoms(
    token_logits: mx.array,
    atom_token_index: mx.array,
    atom_within_token_index: mx.array,
    num_bins: int,
) -> mx.array:
    token_logits = token_logits.reshape(
        token_logits.shape[0], token_logits.shape[1], -1, num_bins
    )
    batch = mx.arange(token_logits.shape[0])[:, None]
    token_idx = atom_token_index
    within = atom_within_token_index
    return token_logits[batch, token_idx, within]


def expectation_from_logits(logits: mx.array, max_value: float) -> mx.array:
    probs = stable_softmax(logits, axis=-1)
    bins = mx.linspace(0.0, max_value, logits.shape[-1])
    return mx.sum(probs * bins, axis=-1)


def sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x)


def silu(x: mx.array) -> mx.array:
    return nn.silu(x)


def normalize_quaternion(q: mx.array, eps: float = 1e-8) -> mx.array:
    norm = mx.sqrt(mx.maximum(mx.sum(q * q, axis=-1, keepdims=True), eps))
    return q / norm


def quaternion_to_matrix(q: mx.array) -> mx.array:
    q = normalize_quaternion(q)
    w, x, y, z = [q[..., i] for i in range(4)]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return mx.stack(
        [
            mx.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            mx.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], axis=-1),
            mx.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], axis=-1),
        ],
        axis=-2,
    )


def random_rotation(batch_size: int) -> mx.array:
    q = mx.random.normal((batch_size, 4))
    return quaternion_to_matrix(q)


def center_random_augmentation(
    coords: mx.array,
    atom_mask: mx.array,
    *,
    centroid_eps: float = 1e-4,
    translation_scale: float = 1.0,
) -> mx.array:
    centroid = masked_mean(coords, atom_mask, axis=1, keepdims=True, eps=centroid_eps)
    centered = coords - centroid
    rot = random_rotation(coords.shape[0])
    rotated = mx.einsum("bij,baj->bai", rot, centered)
    trans = translation_scale * mx.random.normal((coords.shape[0], 1, 3))
    return rotated + trans


def edm_sigmas(
    num_steps: int,
    sigma_data: float,
    s_min: float,
    s_max: float,
    p: float,
) -> mx.array:
    t = mx.linspace(0.0, 1.0, 2 * num_steps + 1)[1::2]
    base = t * (s_min ** (1.0 / p)) + (1.0 - t) * (s_max ** (1.0 / p))
    return sigma_data * (base**p)


def edm_gammas(sigmas: mx.array, s_churn: float, s_tmin: float, s_tmax: float) -> mx.array:
    gamma = mx.full(sigmas.shape, 0.0, dtype=mx.float32)
    active = (sigmas >= s_tmin) & (sigmas <= s_tmax)
    gamma_value = min(s_churn / int(sigmas.shape[0]), math.sqrt(2.0) - 1.0)
    return mx.where(active, mx.full(sigmas.shape, gamma_value, dtype=mx.float32), gamma)
