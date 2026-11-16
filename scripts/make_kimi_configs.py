"""Write the two Kimi-layout configs: Kimi Linear's layer pattern -- three KDA layers then one attention
layer, attention last -- with the attention slots either no-position softmax (the Kimi Linear reference) or dilution.

    PYTHONPATH=src python scripts/make_kimi_configs.py

Everything except the layer list and the FFN width is the published 16K recipe (configs/scale_16384/all_dilution_nope.yaml
of the public DilutionAttention release: 24L x d768 x 12 heads, SwiGLU, context 16,384, 65,536 tokens per step, the
128,174-step cosine, no BOS). KDA layers carry about 142K more parameters than attention layers, so the SwiGLU width is
narrowed to match the published 208,540,416 parameters.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from dilution.config import ModelConfig, build_model

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parents[2] / "dilution-attention/configs/scale_16384/all_dilution_nope.yaml"
TARGET = 208_540_416
ARMS = {
    "kimi_softmax_nope": ("softmax", "Kimi Linear reference: 3 KDA : 1 softmax attention with no position signal, attention last"),
    "kimi_dilution": ("dilution", "Kimi Linear's layer pattern with dilution attention in its attention slots"),
}


def count(model: dict) -> int:
    return sum(p.numel() for p in build_model(ModelConfig(**model)).parameters())


def main() -> int:
    base = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    out = ROOT / "configs/scale_16384"
    out.mkdir(parents=True, exist_ok=True)
    for name, (attn, desc) in ARMS.items():
        cfg = copy.deepcopy(base)
        m = cfg["model"]
        m["attention"] = attn
        m["layer_attention"] = ["kda", "kda", "kda", attn] * 6
        d = m["d_model"]
        m.pop("ffn_hidden", None)
        default_hidden = (8 * d) // 3
        excess = count(m) - TARGET
        m["ffn_hidden"] = int(round(default_hidden - excess / (3 * d * m["n_layers"])))
        params = count(m)
        cfg["training"]["output_dir"] = f"runs/scale_16384/{name}"
        header = (f"# 209M x 16K, {name}: {desc}.\n"
                  f"# Published 16K recipe (public DilutionAttention configs/scale_16384/all_dilution_nope.yaml), no BOS, the\n"
                  f"# 128,174-step cosine run to step 64,000. Parameters {params:,} (published {TARGET:,}; "
                  f"{100 * (params - TARGET) / TARGET:+.3f}%) with SwiGLU width {m['ffn_hidden']}.\n")
        (out / f"{name}.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
        print(f"{name:20s} {params:,} ({100 * (params - TARGET) / TARGET:+.3f}%), ffn_hidden {m['ffn_hidden']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
