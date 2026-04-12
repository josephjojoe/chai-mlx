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

# Experimental kernel for the atom-local 32x128 attention pattern.
# The default code path still prefers mlx.fast.scaled_dot_product_attention.
BLOCKED_LOCAL_ATTENTION_SOURCE = r'''
uint gid = thread_position_in_grid.x;
uint D = q_shape[3];
uint Q = q_shape[2];
uint H = q_shape[1];
uint M = q_shape[0];
uint K = k_shape[2];

uint d = gid % D;
gid /= D;
uint q_idx = gid % Q;
gid /= Q;
uint h = gid % H;
gid /= H;
uint m = gid;

if (m >= M) {
    return;
}

uint q_base = (((m * H + h) * Q + q_idx) * D);
T max_logit = -INFINITY;
for (uint kk = 0; kk < K; ++kk) {
    uint k_base = (((m * H + h) * K + kk) * D);
    T dot = T(0);
    for (uint dd = 0; dd < D; ++dd) {
        dot += q[q_base + dd] * k[k_base + dd];
    }
    T logit = dot * scale + additive_bias[(((m * H + h) * Q + q_idx) * K + kk)];
    max_logit = metal::max(max_logit, logit);
}

T denom = T(0);
for (uint kk = 0; kk < K; ++kk) {
    uint k_base = (((m * H + h) * K + kk) * D);
    T dot = T(0);
    for (uint dd = 0; dd < D; ++dd) {
        dot += q[q_base + dd] * k[k_base + dd];
    }
    T logit = dot * scale + additive_bias[(((m * H + h) * Q + q_idx) * K + kk)];
    denom += metal::exp(logit - max_logit);
}

T acc = T(0);
for (uint kk = 0; kk < K; ++kk) {
    uint k_base = (((m * H + h) * K + kk) * D);
    uint v_base = k_base;
    T dot = T(0);
    for (uint dd = 0; dd < D; ++dd) {
        dot += q[q_base + dd] * k[k_base + dd];
    }
    T logit = dot * scale + additive_bias[(((m * H + h) * Q + q_idx) * K + kk)];
    T w = metal::exp(logit - max_logit) / denom;
    acc += w * v[v_base + d];
}

out[(((m * H + h) * Q + q_idx) * D) + d] = acc;
'''
