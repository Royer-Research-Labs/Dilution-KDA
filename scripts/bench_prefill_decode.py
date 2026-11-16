"""Forward-only (prefill) and cached-decode cost of the fused dilution kernels at long context.

    PYTHONPATH=src python scripts/bench_prefill_decode.py [--lengths 16384,65536,131072,262144]
        [--heads 12] [--what prefill,forward,decode,sdpa] [--budget 16]

Per prompt length (batch 1, bf16, d_head 64) it reports the median time and the peak
allocated GPU memory of:

    prefill  dilution_attention_forward(return_aux="totals"): what begin_cache runs
    forward  dilution_attention_forward(): the no-grad model forward
    decode   dilution_decode_step against a cache of that length
    sdpa     torch scaled_dot_product_attention (causal for the prompt, one row for decode)

One length runs at a time and its buffers are freed before the next. A configuration that
runs out of memory is reported as OOM, not skipped.
"""
from __future__ import annotations

import argparse
import statistics
import sys

import torch
import torch.nn.functional as F

from dilution import kernels as K


def timed(fn, iters, warmup=1):
    """(median ms, peak GiB above what was allocated before the call), or None on OOM."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts), (torch.cuda.max_memory_allocated() - base) / 2**30
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def show(label, T, res, ref=None):
    if res is None:
        print(f"T {T:>8d} | {label:28s}      OOM", flush=True)
        return None
    ms, gib = res
    rel = f"  {ms / ref:5.2f}x sdpa" if ref else ""
    print(f"T {T:>8d} | {label:28s} {ms:10.3f} ms  peak {gib:6.2f} GiB{rel}", flush=True)
    return ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="16384,65536,131072,262144")
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--what", default="sdpa,prefill,forward,decode")
    ap.add_argument("--budget", type=float, default=None, help="scratch_budget_gib override")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--decode-iters", type=int, default=200)
    a = ap.parse_args()
    what = set(a.what.split(","))
    H, D = a.heads, a.head_dim
    kw = {} if a.budget is None else {"scratch_budget_gib": a.budget}
    print(torch.cuda.get_device_name(0), f"| B1 H{H} D{D} bf16 | budget {a.budget or 'default'}")
    for T in (int(x) for x in a.lengths.split(",")):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        iters = a.iters or (9 if T <= 65536 else 3)
        ref = None
        with torch.no_grad():
            if "sdpa" in what:
                ref = show("sdpa causal", T, timed(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), iters))
            if "prefill" in what:
                show("dilution prefill (totals)", T, timed(
                    lambda: K.dilution_attention_forward(q, k, v, dot_precision="bf16", return_aux="totals", **kw),
                    iters), ref)
            if "forward" in what:
                show("dilution forward", T, timed(
                    lambda: K.dilution_attention_forward(q, k, v, dot_precision="bf16", **kw), iters), ref)
            if "decode" in what:
                q1 = q[:, :, -1:].contiguous()
                committed = torch.rand(1, H, T, device="cuda")
                committed[:, :, -1] = 0
                dref = None
                if "sdpa" in what:
                    dref = show("sdpa decode row", T, timed(
                        lambda: F.scaled_dot_product_attention(q1, k, v), a.decode_iters, warmup=20))
                # The step advances `committed` in place; that does not change its cost.
                show("dilution decode step", T, timed(
                    lambda: K.dilution_decode_step(q1, k, v, committed, T), a.decode_iters, warmup=20), dref)
        del q, k, v
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
