"""Kimi Delta Attention (KDA) as a layer type of the DilutionAttention harness.

The other modules of this package are the public harness of https://github.com/Royer-Research-Labs/Dilution-Attention
(vendored.json lists them, with the one recorded edit). That harness imports an optional `mixers` module and registers the
layer types it exports, so `layer_attention: [kda, kda, kda, softmax, ...]` builds the Kimi Linear layer pattern while every
softmax and dilution layer runs exactly the code that produced the published results.

KDA is Moonshot AI's (Kimi Linear, arXiv 2510.26692): a gated delta rule with per-channel decay. This is the
flash-linear-attention parameterisation (fla-core, MIT): q / k / v projections each followed by a short causal depthwise
conv and SiLU, a low-rank per-channel forget gate, a per-head write strength beta, q / k L2-normalised, an RMS-normalised
output gated by a low-rank sigmoid gate. Training and prefill use fla's chunked Triton kernel; the fp32 recurrence below
is the CPU fallback, the reference the kernel is tested against, and the cached-decode step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MIXER_TYPES = ("kda",)


def _fp(x: Tensor) -> Tensor:
    """float64 stays float64 (tests); everything else computes in fp32."""
    return x if x.dtype == torch.float64 else x.float()


def kda_reference(q: Tensor, k: Tensor, v: Tensor, g_raw: Tensor, beta_raw: Tensor, A_log: Tensor,
                  dt_bias: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """fp32 (float64 for float64 inputs) recurrence equal to fla's chunk_kda with the q / k L2 norm, the gate and the
    beta sigmoid in-kernel.

    q, k, v, g_raw: (B, T, H, D); beta_raw: (B, T, H); A_log: (H,); dt_bias: (H * D,); state: (B, H, D, D) in fla's
    [K, V] layout or None. Returns o (B, T, H, D) and the final state, both in the compute dtype."""
    B, T, H, D = q.shape
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g_raw, beta_raw = (t.to(dt) for t in (q, k, v, g_raw, beta_raw))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)   # fla l2norm_fwd, eps 1e-6
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = -A_log.to(dt).exp()[:, None] * F.softplus(g_raw + dt_bias.to(dt).view(H, D))
    beta = beta_raw.sigmoid()
    S = q.new_zeros(B, H, D, D) if state is None else state.to(dt).clone()
    out = torch.empty_like(v)
    scale = D ** -0.5
    for t in range(T):
        S = S * g[:, t].exp()[..., None]
        k_t = k[:, t]
        err = v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, S)
        S = S + torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k_t, err)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S)
    return out, S


@torch.compiler.disable
def _kda_kernel(q, k, v, g, beta, A_log, dt_bias, want_state: bool):
    """fla-core's chunked KDA (Triton), outside the outer torch.compile graph."""
    from fla.ops.kda import chunk_kda
    dtype = v.dtype
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g.to(dtype), beta.to(dtype)))
    return chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, output_final_state=want_state,
                     use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)


class KDALayer(nn.Module):
    """The `kda` layer type: Kimi Delta Attention with fla's parameterisation, heads = model heads."""

    conv_size = 4

    def __init__(self, config):
        super().__init__()
        C, H = config.d_model, config.n_heads
        D = C // H
        self.n_heads, self.head_dim = H, D
        self.q_proj = nn.Linear(C, H * D, bias=False)
        self.k_proj = nn.Linear(C, H * D, bias=False)
        self.v_proj = nn.Linear(C, H * D, bias=False)
        self.q_conv, self.k_conv, self.v_conv = (
            nn.Conv1d(H * D, H * D, self.conv_size, groups=H * D, bias=False) for _ in range(3))
        for conv in (self.q_conv, self.k_conv, self.v_conv):
            nn.init.normal_(conv.weight, std=0.02)
        self.f_proj = nn.Sequential(nn.Linear(C, D, bias=False), nn.Linear(D, H * D, bias=False))
        self.b_proj = nn.Linear(C, H, bias=False)
        self.A_log = nn.Parameter(torch.empty(H).uniform_(1, 16).log())
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(H * D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(C, D, bias=False), nn.Linear(D, H * D, bias=True))
        self.o_norm_weight = nn.Parameter(torch.ones(D))
        self.o_proj = nn.Linear(H * D, C, bias=False)
        self.use_kernel = True

    def _conv(self, conv: nn.Conv1d, z: Tensor, hist: Tensor | None):
        """SiLU(causal depthwise conv) over the earlier kernel - 1 inputs (zeros before the first token)."""
        if hist is None:
            hist = z.new_zeros(z.shape[0], self.conv_size - 1, z.shape[2])
        full = torch.cat([hist.to(z.dtype), z], dim=1)
        y = conv(full.transpose(1, 2)).transpose(1, 2)
        return F.silu(y), full[:, -(self.conv_size - 1):]

    def _features(self, x: Tensor, conv_state=None):
        B, T, _ = x.shape
        hq, hk, hv = (None, None, None) if conv_state is None else conv_state
        q, hq = self._conv(self.q_conv, self.q_proj(x), hq)
        k, hk = self._conv(self.k_conv, self.k_proj(x), hk)
        v, hv = self._conv(self.v_conv, self.v_proj(x), hv)
        q, k, v = (t.reshape(B, T, self.n_heads, self.head_dim) for t in (q, k, v))
        g = self.f_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        return q, k, v, g, self.b_proj(x), (hq, hk, hv)

    def _kernel_ok(self, q: Tensor) -> bool:
        return self.use_kernel and q.is_cuda

    def _recur(self, q, k, v, g, beta, want_state: bool):
        if self._kernel_ok(q):
            return _kda_kernel(q, k, v, g, beta, self.A_log, self.dt_bias, want_state)
        o, state = kda_reference(q, k, v, g, beta, self.A_log, self.dt_bias)
        return o.to(v.dtype), state

    def _out(self, o: Tensor, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        gate = self.g_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        o = _fp(o)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5) * self.o_norm_weight.to(o.dtype)
        o = (o * gate.to(o.dtype).sigmoid()).to(gate.dtype)
        return self.o_proj(o.reshape(B, T, -1))

    def forward(self, x: Tensor) -> Tensor:
        q, k, v, g, beta, _ = self._features(x)
        o, _ = self._recur(q, k, v, g, beta, want_state=False)
        return self._out(o, x)

    def begin_cache(self, x: Tensor, capacity: int | None = None):
        q, k, v, g, beta, conv_state = self._features(x)
        o, state = self._recur(q, k, v, g, beta, want_state=True)
        return self._out(o, x), {"S": _fp(state), "conv": conv_state}

    def step_cache(self, x1: Tensor, cache: dict) -> Tensor:
        q, k, v, g, beta, conv_state = self._features(x1, cache["conv"])
        o, state = kda_reference(q, k, v, g, beta, self.A_log, self.dt_bias, cache["S"])
        cache["S"], cache["conv"] = state, conv_state
        return self._out(o, x1)


MIXER_CLASSES = {"kda": KDALayer}
