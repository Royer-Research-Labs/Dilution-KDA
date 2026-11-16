"""Render the README figures for the Kimi pair (209M, 16K training context, 3 seeds per arm).

    PYTHONPATH=src python scripts/make_kimi_figures.py [--outdir docs/figures]

kimi-retrieval.png  NIAH accuracy by context, one row per model and seed (extrapolation range varies between seeds,
                    so every seed is shown rather than a mean). Cells up to 16K come from the 20-placement in-context
                    ladder (results/niah/scale1x__*), longer ones from the 10-placement ladders (extrap*__*). The
                    reference was not run past 128K.
kimi-pg19.png       PG-19 loss by position range (results/longppl/pg19_kimi_s*.json), thin lines per seed, thick
                    line the seed mean.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1337, 2357, 7331)
ARMS = (("kimi_softmax_nope", "Kimi reference (softmax)"), ("kimi_dilution", "Kimi + dilution"))
COLOR = {"kimi_softmax_nope": "#8a8f98", "kimi_dilution": "#2463c9"}
COLS = (1024, 3072, 6144, 12288, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 192000, 256000)
TRAINED = 16384


def niah(arm: str, seed: int) -> dict[int, float]:
    acc = {}
    for f in sorted((ROOT / "results/niah").glob(f"*__{arm}_seed{seed}.json")):
        if f.name.startswith("kept"):
            continue
        for r in json.loads(f.read_text()):
            acc[r["context_length"]] = r["accuracy"]
    return acc


def label(n: int, base: int = 1024) -> str:
    return f"{round(n / base)}K" if n >= 1000 else str(n)


def retrieval(out: Path) -> None:
    rows, names = [], []
    for arm, name in ARMS:
        for s in SEEDS:
            a = niah(arm, s)
            rows.append([a.get(c, np.nan) for c in COLS])
            names.append(f"{name}  seed {s}")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f0f0f0")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt, col = ("n/r", "#999999") if np.isnan(v) else (f"{v:.2f}", "white" if v > 0.6 else "#222222")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(len(COLS)), [label(c, 1000) for c in COLS], fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    edge = sum(c <= TRAINED for c in COLS) - 0.5
    ax.axvline(edge, color="#d9822b", lw=2)
    ax.axhline(2.5, color="white", lw=3)
    ax.set_xlabel("context length (tokens); orange line = trained context (16K)")
    ax.set_title("Needle-in-a-haystack accuracy, 209M, trained at 16K (n/r = not run)", fontsize=10.5, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def pg19(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ranges = None
    for arm, name in ARMS:
        per_seed = []
        for s in SEEDS:
            d = json.loads((ROOT / f"results/longppl/pg19_kimi_s{s}.json").read_text())
            ranges = d["ranges"]
            pb = d["models"][f"{arm}_s{s}"]["passes"]["plain"]["per_book"]
            per_seed.append([st.mean(b[i] for b in pb) for i in range(len(ranges))])
        x = np.arange(len(ranges))
        for y in per_seed:
            ax.plot(x, y, color=COLOR[arm], lw=1, alpha=0.35)
        mean = np.mean(per_seed, axis=0)
        ax.plot(x, mean, color=COLOR[arm], lw=2.2, marker="o", ms=4, label=name)
        ax.text(x[-1] + 0.15, mean[-1], name, color="#333333", fontsize=8.5, va="center")
    ax.set_xticks(x, [f"{label(lo) if lo else 0}-{label(hi)}" for lo, hi in ranges], rotation=35, fontsize=8.5)
    edge = sum(hi <= TRAINED for _, hi in ranges) - 0.5
    ax.axvline(edge, color="#d9822b", lw=1.5, ls="--")
    ax.text(edge + 0.1, ax.get_ylim()[1], " trained context", color="#d9822b", fontsize=8.5, va="top")
    ax.set_xlim(-0.3, len(ranges) + 1.6)
    ax.set_ylabel("loss (nats / token)")
    ax.set_xlabel("token position in the book")
    ax.set_title("PG-19 loss by position (29 books to 128K tokens; thin lines: seeds)", fontsize=10.5, loc="left")
    ax.grid(axis="y", color="#e6e6e6")
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "docs/figures"))
    out = Path(ap.parse_args().outdir)
    out.mkdir(parents=True, exist_ok=True)
    retrieval(out / "kimi-retrieval.png")
    pg19(out / "kimi-pg19.png")
    print(f"wrote {out}/kimi-retrieval.png, {out}/kimi-pg19.png")


if __name__ == "__main__":
    main()
