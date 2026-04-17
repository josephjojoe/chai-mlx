from __future__ import annotations

SWIGLU_SOURCE = r'''
// out = SiLU(u[..., :D]) * u[..., D:] where D = u_last_dim / 2.
// MLX auto-injects `u_shape`/`u_ndim` for the input; output shape/ndim are
// not available, so derive D from the input. Compute in fp32 for bf16 safety
// (metal::exp returns float; implicit bfloat conversion is disallowed).
uint elem = thread_position_in_grid.x;
uint d2 = u_shape[u_ndim - 1];
uint d = d2 / 2;
uint outer = elem / d;
uint inner = elem % d;
uint base = outer * d2 + inner;
float a = float(u[base]);
float b = float(u[base + d]);
float silu_a = a / (1.0f + metal::exp(-a));
out[elem] = T(silu_a * b);
'''

GATED_RESIDUAL_SOURCE = r'''
// Fused: out = x + sigmoid(gate) * sub, computed in fp32 for numerical
// stability and to avoid bfloat16 implicit conversion issues.
uint elem = thread_position_in_grid.x;
float g = 1.0f / (1.0f + metal::exp(-float(gate[elem])));
out[elem] = T(float(x[elem]) + g * float(sub[elem]));
'''

ADALN_APPLY_SOURCE = r'''
uint elem = thread_position_in_grid.x;
out[elem] = x_norm[elem] * (T(1) + scale[elem]) + shift[elem];
'''

FUSED_ADALN_SOURCE = r'''
// One threadgroup per row: computes LayerNorm + conditional affine in a
// single pass so the normalized intermediate is never written to device memory.
//
// Inputs:  x[rows, D], ln_w[D], ln_b[D], scale[rows, D], shift[rows, D], eps (scalar)
// Output:  y[rows, D]  =  LN(x) * (1 + scale) + shift
//
// grid      = (1, num_rows, 1)
// threadgrp = (tg_size, 1, 1)
// MLX injects `thread_index_in_threadgroup` as a scalar uint, but
// `thread_position_in_grid` and `threads_per_threadgroup` as uint3,
// so we must use `.x` on the latter two but not on the former.

uint row = thread_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint tg  = threads_per_threadgroup.x;

uint D = x_shape[x_ndim - 1];
uint rows = 1;
for (uint i = 0; i < x_ndim - 1; i++) rows *= x_shape[i];
if (row >= rows) return;

uint base = row * D;

// --- pass 1: mean ---
float local_sum = 0.0f;
for (uint d = tid; d < D; d += tg)
    local_sum += float(x[base + d]);

// Warp-level reduce (simd_sum covers a full SIMD group = 32 threads on Apple GPU).
local_sum = simd_sum(local_sum);

// Cross-warp reduce via threadgroup memory.
constexpr uint MAX_WARPS = 32;
threadgroup float warp_buf[MAX_WARPS];
uint warp_id  = tid / 32;
uint lane_id  = tid % 32;
uint n_warps  = (tg + 31) / 32;

if (lane_id == 0) warp_buf[warp_id] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

float total = 0.0f;
if (tid < n_warps) total = warp_buf[tid];
total = simd_sum(total);

float mean = total / float(D);

// --- pass 2: variance ---
float local_var = 0.0f;
for (uint d = tid; d < D; d += tg) {
    float diff = float(x[base + d]) - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (lane_id == 0) warp_buf[warp_id] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);

float var_total = 0.0f;
if (tid < n_warps) var_total = warp_buf[tid];
var_total = simd_sum(var_total);

float inv_std = metal::rsqrt(var_total / float(D) + float(eps));

// --- pass 3: normalize + affine ---
for (uint d = tid; d < D; d += tg) {
    float x_norm = (float(x[base + d]) - mean) * inv_std;
    x_norm = x_norm * float(ln_w[d]) + float(ln_b[d]);
    y[base + d] = T(x_norm * (1.0f + float(scale[base + d])) + float(shift[base + d]));
}
'''


FUSED_ADALN_NOAFFINE_SOURCE = r'''
// Same as FUSED_ADALN_SOURCE but without the LayerNorm affine (weight=1, bias=0).
// Used by AdaLayerNorm, whose inner LN is affine=False. Skipping the affine
// avoids materializing dummy ones/zeros buffers of size D per layer.
//
// Inputs:  x[rows, D], scale[rows, D], shift[rows, D], eps (scalar)
// Output:  y[rows, D]  =  ((x - mean) / sqrt(var + eps)) * (1 + scale) + shift

uint row = thread_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint tg  = threads_per_threadgroup.x;

uint D = x_shape[x_ndim - 1];
uint rows = 1;
for (uint i = 0; i < x_ndim - 1; i++) rows *= x_shape[i];
if (row >= rows) return;

uint base = row * D;

float local_sum = 0.0f;
for (uint d = tid; d < D; d += tg)
    local_sum += float(x[base + d]);
local_sum = simd_sum(local_sum);

constexpr uint MAX_WARPS = 32;
threadgroup float warp_buf[MAX_WARPS];
uint warp_id  = tid / 32;
uint lane_id  = tid % 32;
uint n_warps  = (tg + 31) / 32;

if (lane_id == 0) warp_buf[warp_id] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

float total = 0.0f;
if (tid < n_warps) total = warp_buf[tid];
total = simd_sum(total);

float mean = total / float(D);

float local_var = 0.0f;
for (uint d = tid; d < D; d += tg) {
    float diff = float(x[base + d]) - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (lane_id == 0) warp_buf[warp_id] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);

float var_total = 0.0f;
if (tid < n_warps) var_total = warp_buf[tid];
var_total = simd_sum(var_total);

float inv_std = metal::rsqrt(var_total / float(D) + float(eps));

for (uint d = tid; d < D; d += tg) {
    float x_norm = (float(x[base + d]) - mean) * inv_std;
    y[base + d] = T(x_norm * (1.0f + float(scale[base + d])) + float(shift[base + d]));
}
'''
