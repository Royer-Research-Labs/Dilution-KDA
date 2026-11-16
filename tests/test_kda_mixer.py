"""The KDA layer type and the vendored harness (CPU; the kernel test needs CUDA and fla-core)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from dilution.config import ATTENTION_TYPES, ModelConfig, build_model
from dilution.mixers import KDALayer, kda_reference

ROOT = Path(__file__).resolve().parents[1]


def _tiny(attn: str) -> ModelConfig:
    return ModelConfig(vocab_size=97, n_layers=8, d_model=64, n_heads=2, context_length=64, bias=False, mlp="swiglu",
                       positional="none", attention=attn, layer_attention=["kda", "kda", "kda", attn] * 2)


def test_kda_is_registered():
    assert "kda" in ATTENTION_TYPES
    m = build_model(_tiny("softmax"))
    assert sum(isinstance(b.attn, KDALayer) for b in m.transformer.h) == 6


@pytest.mark.parametrize("attn", ["softmax", "dilution"])
def test_cached_decode_matches_forward(attn):
    """Kimi layout (3 KDA : 1 attention, attention last): prefill + cached decode equal the full forward on CPU."""
    torch.manual_seed(0)
    m = build_model(_tiny(attn)).float().eval()
    x = torch.randint(0, 97, (2, 40))
    with torch.no_grad():
        full, _ = m(x, x)
        logits, caches = m.prefill(x[:, :24], capacity=48)
        torch.testing.assert_close(logits[:, -1], full[:, 23], rtol=1e-4, atol=1e-4)
        for t in range(24, 40):
            torch.testing.assert_close(m.decode_step(x[:, t:t + 1], t, caches)[:, -1], full[:, t], rtol=1e-4, atol=1e-4)


def test_decay_parameters_are_tagged_and_exempt():
    """A_log and dt_bias carry _no_weight_decay; train.py puts exactly those in a zero-decay group."""
    m = build_model(_tiny("dilution"))
    tagged = [n for n, p in m.named_parameters() if getattr(p, "_no_weight_decay", False)]
    assert tagged and all(n.endswith(("A_log", "dt_bias")) for n in tagged)
    assert len(tagged) == 2 * 6
    src = (ROOT / "src/dilution/train.py").read_text(encoding="utf-8")
    assert '{"params": undecayed, "weight_decay": 0.0}' in src


def test_reference_matches_a_stepwise_recurrence_in_float64():
    """Chunk-free check of kda_reference on a 2-token sequence against the delta rule written out by hand."""
    torch.manual_seed(1)
    B, T, H, D = 1, 2, 1, 4
    q, k, v, g = (torch.randn(B, T, H, D, dtype=torch.float64) for _ in range(4))
    beta = torch.randn(B, T, H, dtype=torch.float64)
    A_log, dt_bias = torch.zeros(H, dtype=torch.float64), torch.zeros(H * D, dtype=torch.float64)
    o, _ = kda_reference(q, k, v, g, beta, A_log, dt_bias)
    qn = q / q.norm(dim=-1, keepdim=True)
    kn = k / k.norm(dim=-1, keepdim=True)
    decay = (-torch.nn.functional.softplus(g)).exp()          # A = exp(0) = 1
    S = torch.zeros(D, D, dtype=torch.float64)
    for t in range(T):
        S = S * decay[0, t, 0][:, None]
        S = S + torch.outer(kn[0, t, 0], (v[0, t, 0] - kn[0, t, 0] @ S)) * beta[0, t, 0].sigmoid()
        torch.testing.assert_close(o[0, t, 0], (qn[0, t, 0] * D ** -0.5) @ S, rtol=1e-6, atol=1e-6)


def test_vendored_files_match_their_record():
    rec = json.loads((ROOT / "vendored.json").read_text(encoding="utf-8"))
    for f in rec["files"]:
        assert hashlib.sha256((ROOT / f["file"]).read_bytes()).hexdigest() == f["sha256"], f["file"]
    assert [f["file"] for f in rec["files"] if f["edits"]] == ["src/dilution/train.py"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and fla-core")
def test_fla_kernel_matches_reference():
    torch.manual_seed(2)
    layer = KDALayer(_tiny("softmax")).cuda()
    x = torch.randn(2, 128, 64, device="cuda")
    q, k, v, g, beta, _ = layer._features(x)
    o_k, s_k = layer._recur(q, k, v, g, beta, want_state=True)
    o_r, s_r = kda_reference(q, k, v, g, beta, layer.A_log, layer.dt_bias)
    torch.testing.assert_close(o_k.float(), o_r.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(s_k.float(), s_r.float(), rtol=2e-3, atol=2e-3)
