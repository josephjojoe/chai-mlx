from __future__ import annotations

SWIGLU_SOURCE = r'''
uint elem = thread_position_in_grid.x;
uint d = out_shape[out_ndim - 1];
uint outer = elem / d;
uint inner = elem % d;
uint base = outer * (2 * d) + inner;
T a = u[base];
T b = u[base + d];
out[elem] = (a / (T(1) + metal::exp(-a))) * b;
'''

GATED_RESIDUAL_SOURCE = r'''
uint elem = thread_position_in_grid.x;
T gate_val = T(1) / (T(1) + metal::exp(-gate[elem]));
out[elem] = x[elem] + gate_val * sub[elem];
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

uint row = thread_position_in_grid.y;
uint tid = thread_index_in_threadgroup.x;
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
