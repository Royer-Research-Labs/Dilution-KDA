# Fused kernels

`src/dilution/kernels.py`. The exclusive column-cumsum is the operator's only
sequential coupling. It is either carried down the query blocks by programs
that each own some keys (training, and forward-only calls from 4K tokens), or
decomposed into per-row-block partial key-sums joined by one fused Triton block
scan (short calls). This document describes the current design and its measured
cost; how the training kernels got here, what was tried and the full-schedule
validation are summarised in S26.

## Training forward (two score sweeps)

    K1 (grid N x row-blocks): online causal-softmax row statistics, with
        pk = SCALE * p @ k accumulated during that same statistics sweep
    KF (grid N x key-blocks; query blocks in causal order): reconstruct p,
        retain per-key cumulative demand in registers, save incoming carry
        checkpoints, and accumulate share @ V and row share sums with atomics
    normalize: out = accumulated numerator / row share sum

KF replaces a second score sweep, the block scan, and a separate output pass.
It stores the carry checkpoints that the full-gradient backward needs, but does
not allocate the block-sum buffer S. The checkpoints come every 32 rows
(`BWD_BLOCK_M`), which is the tile height the backward's BA pass needs. Up to 4K
tokens KF runs on those 32-row tiles (K1 stays at 64). From `KF_TALL_MIN_SEQ` =
4096 it runs on 64-row tiles (`KF_BLOCK_M`) and reads the checkpoint inside each
tile off its scan, as the largest prefix sum over the first 32 rows: 1.7% of a
step at 4K, 2.3% at 8K and 2.6% at 16K, against a 3% loss at B32 H12 T1024. Its
output/row-sum atomics use relaxed ordering; only the next kernel reads their
completed sums.
Moving pk into K1's online sweep is necessary for this to pay: calculating it
with more atomics in KF lost the saving. The online accumulator is rescaled
whenever the running row maximum changes, as is the softmax denominator.

A training step makes four score products. Full cumulative demand and its
backward suffix remain; no gradient is detached.

## Tile bodies

The sweeps of the training step and of the forward-only path share three
choices. They came from counting instructions: a warp ran about 1,700
instructions per 64x64 tile of the share sweep and only 64 of them were
tensor-core products (see "Where the time goes").

- **Unmasked interior tiles.** The causal / length mask only matters on the
  tiles the diagonal crosses and on a partial last block or key tile. Each
  kernel's loop is split into those and the whole tiles below the diagonal,
  whose body is compiled without the mask (`_bids`, `MASKED`): no index
  compares, no selects, unmasked loads and stores, and the scale and row-max
  subtraction contract into one FMA. A masked tile applies the mask once, to the
  affinities, which makes p exactly zero there and with it every product formed
  from p; nothing downstream is masked again. Statistics-only K1 launches split
  their sweep the same way. The row-parallel K1 -> scan -> K2 path does not: it
  keeps one masked loop, and `kernels_bid.py` is pinned bitwise to it.
- **Reciprocal.** Triton's `/` is PTX `div.full`: a reciprocal wrapped in range
  checks and rescaling, 8-10 instructions per element. The sums the kernels
  divide by are positive and at least eps, so `_rcp` emits `rcp.approx.ftz.f32`
  (one instruction, at most 1 ulp; within 2 ulp of true division over
  [1e-9, 1e6] in tests/test_kernels.py) and the share is `p * _rcp(sum)`. This is
  inline PTX, so these kernels are NVIDIA-only.
- **Column sums from the scan.** The running sums advance by the column total
  of p. A separate `tl.sum(p, axis=0)` reduces across warps in the tensor-core
  layout (about 150 instructions and a barrier). p is non-negative, so the total
  is the largest prefix sum, and `tl.max` of the scan's own output is reduced in
  the scan's layout for a fraction of that.

On the 64K share sweep (53.1 ms) these give 1.10x, 1.05x and 1.07x alone and
1.30x together (40.8 ms); the tile shape below takes it to 37.3 ms. Outputs
change at rounding level (the FMA and the reduction order), not bitwise.

## Forward-only and prefill (two score sweeps, key spans)

From `KEY_SPAN_MIN_SEQ` = 4096 tokens, a call that does not need the backward's
carry runs:

    K1 (grid N x row-blocks, statistics only): causal-softmax row stats
    KS (grid N x key-spans; query blocks in causal order): each program owns
        a span of key tiles. Per query block it rebuilds p for its tiles,
        continues their running column sums, forms share = p / (sum + eps),
        and adds the block's share @ V and share row-sum to the output once
    normalize: out = accumulated numerator / row share sum

Memory is O(N T): the running column sums live in the `(N, T)` totals buffer,
which each program reads and writes only for its own keys and which ends as the
decode accumulator. Nothing quadratic is allocated, so there is no scratch
budget on this path. `return_aux="totals"` is the prefill form of the call (the
row stats and totals without a carry); the model's `begin_cache` uses it.

Both sweeps run on tiles of 128 rows x 32 keys (4 warps, 2 stages). With the
slimmed bodies that shape is the best of 60 for the share sweep at 64K
(37.3 ms; 64 x 64 is 41.9, 64 x 32 with 2 warps 42.3, 128 x 16 44.6, 256 x 32
with 8 warps 45.2, 32-row tiles 52 and up) and within 5% of the best of 96 for
the statistics sweep (11.1 ms against 15.3 at 64 x 64 unsplit; 256 x 64 reaches
10.6 at 246 registers). Where the 128-row query tile would not leave shared
memory for pipelined key / value tiles (fp32 operands wider than 64, bf16 wider
than 128) the tiles are 64 x 32 with one stage (`_key_span_tile`).

A span adds a query block's output once instead of once per tile, but re-reads
its key and value tiles and column sums from memory for every query block, and
the program that owns the first keys visits every row, so wide spans leave the
GPU idle at the end of the launch. The span is the widest power-of-two number
of tiles, up to `KEY_SPAN_MAX_KEYS` = 512 keys, that still leaves
`KEY_SPAN_TARGET_PROGRAMS` = 768 programs (at 12 heads: 64 keys at 4K tokens,
256 at 16K, 512 from 32K). A one-tile span is compiled as the training forward
without its checkpoints: key tile, value tile and sums stay in registers, which
wins when there are too few programs for spans (B1 H1 T16384: 0.43 ms against
0.61 for two-tile spans). Second sweep alone, B1 H12, ms:

    T        one tile,     keys per span
             registers       64     128     256     512    1024
    4096        0.27        0.25    0.29    0.40    0.74    1.28
    16384       4.04        3.17    2.67    2.39    3.45    5.58
    65536      75.6        55.5    43.9    38.6    37.7    40.5
    131072    347         228     180     157     148     157

Outputs match the reference to the same tolerance as the row-parallel path. In
bf16 or tf32 mode two paths can differ by more than fp32 rounding (up to 2e-3
absolute at 32K): a last-bit change in an fp32 share can move its rounding to
the operand precision in the share @ V product. With exact (`ieee`) products the
key-span and row-parallel outputs agree to 2e-5 at 12K tokens. Like training,
this path accumulates rows with atomics and is not bitwise deterministic.

### Row-parallel path (short calls, and whenever the carry is returned)

    K1 (grid N x nB): causal-softmax row stats (rmax, rden) + per-block
        key-sums S of p
        carry = exclusive block-cumsum(S)           # Triton, (N, nB, T)
    K2 (grid N x nB): rebuild p from the stats, committed = carry + in-block
        exclusive cumsum(p), share = p / (committed + p + eps)
        -> row-normalize -> @ V

Nothing (T x T)-shaped reaches DRAM; scratch is O(nB * T). Tiles are loaded
in their native dtype (bf16 under autocast) and the q.k / p.v products run on
bf16 tensor cores with fp32 accumulation (`dot_precision="bf16"`, the
default); the cumsum/share math stays fp32. `return_aux=True` also hands back
the inclusive column totals of p, which are the decode accumulator after the
prefix, for free from the scan buffer.

The scan also reduces the inclusive column totals in the same launch. It
masks K1's unwritten future-key triangle, so S needs no zero fill; there is no
separate full-size cumsum temporary or totals reduction. This path has no
atomics and no normalization launch, and makes three score sweeps.

When the carry is not returned, the scratch is capped at `scratch_budget_gib`
(default 4): the queries are processed in chunks of row blocks, and the scan of
each chunk continues from the column totals of the chunks before it. If one row
block of every (batch, head) row does not fit, the rows are chunked as well.
This is what made long prefill linear in memory before the key-span path
existed, and it still bounds short calls with very many rows. Measured on the
row-parallel path alone (B1 H12): 357 ms / 4.4 GiB at 128K and 1434 ms / 4.8 GiB
at 256K, against 368 ms / 24.4 GiB and 3100 ms / 96.8 GiB unchunked (the latter
spilling into shared system memory), and 25.4 s / 7.1 GiB at 1M, which could
not run. At 256K the budget costs nothing down to 4 GiB (16 / 4 / 1 / 0.25 GiB:
1460 / 1418 / 1514 / 3374 ms); 0.25 GiB leaves 120 programs per launch.

Training sequences of at most `KEY_FORWARD_MIN_SEQ` = 256 tokens also use this
decomposition, with online pk enabled. This path saves its carry every 64 rows,
so its backward runs on 64-row tiles, which spill heavily; the threshold used to
be 1024, and moving the 209M 1K run onto the key-parallel path took it from 88K to
110K tok/s (S26). With the current backward, key-parallel over row-parallel time
is 1.01 / 1.00 / 0.95 / 0.86 at 128 / 256 / 512 / 1024 tokens
(`scripts/bench_crossover.py`), so the threshold is 256. Outputs and gradients
of the two paths agree to bf16 rounding (at most 1.8e-4 relative).

## Backward

Let ds be the gradient at the share (from the row-normalize and the p.v
product), inv = 1/(committed + p + eps), and dk_share = -ds * p * inv^2.
The gradient reaching p_ij is dp_ij = ds (committed + eps) inv^2 + suffix_ij,
where suffix_ij is the sum of dk_share over every *later* query on the same
key. The softmax Jacobian then gives da = p (dp - ghat) with ghat_i =
sum_j p_ij dp_ij. Two facts make this two kernels: the suffix is a running sum
if row blocks are visited last to first, and the -ghat term is linear, so it
can be applied after the fact:

    BA  (grid N x key-blocks; 32-row blocks in REVERSE order): recompute p
        from the saved row stats, ds from do.v, dk_share; keep a running
        per-key suffix sum so dp is exact; accumulate
            dv[j]      += sum_i attn_ij do_i               (in registers)
            dk_main[j] += SCALE sum_i (p dp)_ij q_i        (in registers)
            dq_main[i] += SCALE sum_j (p dp)_ij k_j        (fp32 atomics, relaxed)
            ghat[i]    += sum_j (p dp)_ij                  (64-wide atomics, relaxed)
    BC  (grid N x key-blocks, 64-row blocks): the Jacobian correction
        dk[j] = dk_main[j] - SCALE sum_i p_ij ghat_i q_i. Needs only p (row
        stats): no carry, no cumsum, no do tile
    epilogue kernel: dq = dq_main - ghat * pk, stored in the gradient dtype;
        BA stores dv and BC the corrected dk in the gradient dtype directly

BA does not form committed or inv^2 as tiles. With C = committed + p + eps,
inv = 1/C and share = p inv, the two expressions above are

    dk_share = -(ds share) inv
    p dp     = ds share (1 - share) + p suffix        since (committed + eps) inv = 1 - share

which is 4 multiplies and 2 FMAs per element in place of 8 multiplies and 5
adds. With the unmasked interior body and `_rcp` (see "Tile bodies") BA's loop
went from 1,660 to 1,015 instructions per warp per tile and from 16.3 to 13.4 ms
at the 209M 16K shape. Gradients against autograd through the reference are
unchanged (dq / dk / dv 2.0e-5 / 1.4e-5 / 2.6e-6 relative at 2,500 tokens before
and after, exact products).

dv is accumulated in BA from the same attn tile. At 64-row tiles that extra
product doubles BA (255 registers, spills 46 -> 200), whether dv is a register
accumulator or per-tile atomics; at 32-row tiles BA fits (26 spills, 10 with the
slimmed body) and the product costs 0.45 ms. Forming attn before ds keeps the
two tiles from being live together. 16-row tiles lose to iteration count, and
64-row tiles still spill after the slimming (24.1 against 13.6 ms at 16K).

BA loads the next row block's q and do tiles one iteration ahead and runs with
num_stages=2 on the bf16 path, so the remaining per-row loads (row stats,
carry) are pipelined as well. Neither change helps alone; together BA drops
from 3.14 to 2.82 ms at B8 H8 T4096 and from 18.1 to 16.2 ms at the 209M 16K
shape (B2 H12). Prefetching the row stats and carry by hand as well is slower
(42 spills), as are register caps (`maxnreg` 128-200), 8 warps, and 128-key
tiles.

The epilogue reads pk saved by K1. Moving that product out of BA removes one
tensor-core product and one atomic accumulation per tile from the most
register-constrained pass. It increases K1's time, but saves more in BA.

BA reads `dout` in fp32: a bf16 copy made its compiled schedule 0.6 ms slower
before the prefetch and is neutral with it (2.84 vs 2.82 ms), so it stays
fp32. BC does not read `dout`.

Both kernels recompute tiles from the forward's saved stats (rmax, rden,
carry, ssum) plus dbar_i = do_i . out_i. Exposed as
`dilution.kernels.dilution_attention` (torch.autograd.Function). Gradient
parity vs autograd through the reference on dq, dk, dv at <= 2e-3 relative
in fp32 mode and <= 1e-1 relative in bf16-operand mode (tests/test_kernels.py).

The atomics use relaxed memory ordering: nothing in-kernel reads the
accumulated buffers and the launch boundary is a full sync. The default
acquire-release ordering cost 2.4 ms per step at B8 H8 T4096.

An earlier four-kernel backward (row-parallel key-sums and partial ghat, a
reverse block scan, then separate ghat, dv and dk/dq kernels) is still defined
in kernels.py but is not launched.

## Cached decode

Decode state per dilution layer: the KV cache plus **one** fp32 accumulator
vector (committed column totals of p). Prefill obtains it free from the
forward kernel's scan buffer. Each token step is one fused kernel
(`dilution_decode_step`, grid = batch x heads):

    sweep 1: affinities q.k over the cached keys -> O(T) fp32 scratch;
             online max/sum
    sweep 2: p from the stats, share = p/(committed+p+eps), sum(share . v)
             and sum(share); committed += p stored in place

K is streamed once, V once, the scratch once; output = acc / sum(share) is
stored in the activation dtype. Caches are preallocated (at the context length,
or at prompt + new tokens when generating past it) and filled by slice
assignment, so a step never reallocates.

One program per (batch, head) row streams the whole cache on one SM: 12 of
about 188 SMs at batch 1 with 12 heads. From `DECODE_SPLIT_MIN_LEN` = 8192
cached keys the step is split over the keys instead (grid = rows x chunks of
`DECODE_CHUNK` = 1024 keys, at most 256 chunks, so chunks grow past 256K):

    stats (rows x chunks): affinities -> scratch; the chunk's own max / sum
    share (rows x chunks): merge the chunk statistics into the row's
             normaliser, then p, shares, committed += p, and this chunk's
             part of sum(share . v) and sum(share)
    merge (rows): sum the parts in a fixed order, normalise, store

Two passes over the keys are needed either way: the shares need the final row
normaliser, and `committed += p` must use the final p. The parts are merged in
chunk order, not with atomics, so a step is deterministic. Step time, B1 H12
D64 bf16, ms (the softmax row is `scaled_dot_product_attention` on one query):

    cached keys      16K     64K     128K    256K     1M
    one program     0.149   0.762   1.484   2.959   11.86
    split           0.071   0.168   0.302   0.569    2.31
    softmax row     0.038   0.162   0.279   0.533    1.98

From 64K the split step is within 4-9% of the softmax row, which is memory
bandwidth: both stream K and V once. Below 8K the three launches cost more than
they save (4K: 0.069 against 0.059 ms). At B8 H12 the single-program form
already fills the GPU and the split is 1.04-1.13x faster from 4K. Chunk sizes
of 512-2048 keys are equivalent at 128K; 4096 and up lose.

## Long context (S14, S15)

Three limits surface between 64K and 256K tokens, all in addressing rather
than arithmetic:

- **64-bit scratch offsets.** The carry/S buffers are `(n, T/BLOCK_M, T)` fp32,
  so `n * nB * T` passes 2^31 at 12 heads and T = 131,072: the per-head stride
  is 2.7e8 and the last head's base is 2.95e9, which wraps int32 and raises an
  illegal memory access. Only the per-program base multiply is promoted
  (`pid_n.to(tl.int64) * stride_sn`), hoisted out of the row-block loops so the
  tile offsets stay 32-bit. In-step cost at B8 H8 T4096: 5.78 ms against
  5.76-5.80 unpatched, i.e. free.
- **Forward scratch.** The row-parallel scratch is quadratic in T -- 24 GiB at
  H=12 T=128K and 96 GiB at 256K, which no longer fits. It was first bounded by
  processing the (batch, head) rows in chunks (256K at B1 H12 D64 in 17.9 GiB
  and 1.8 s), which still left prefill unbounded because prefill asked for the
  aux, and could not reach 1M tokens (128 GiB for a single row). Forward-only
  calls and prefill now take the key-span path above, which has no quadratic
  scratch: **256K in 0.8 GiB and 0.77 s, 1M in 3.2 GiB and 13.6 s**.
- **Grid dimension y.** CUDA caps grid dim 1 at 65,535, and the block scan
  launches `T / KEYS` programs on it -- 65,536 at T = 256K with the KEYS = 4
  tile the shared-memory budget forces at that block count. The scan's grid
  axes are swapped so key tiles ride axis 0.

Inference also skips the autograd Function entirely (`torch.is_grad_enabled()`
is false), so no backward carry is allocated at all, and the NIAH loader turns
the fused kernels on: the eager tensor path materialises a (T x T) matrix per
head and needs ~12 GiB per layer at 16K, which OOMs a 24-layer eval.

## Ablation kernels (`kernels_bid.py`)

`SHARE` is a three-way constexpr: 0 = no share step (`attn = p`, the softmax
control), 1 = inclusive `p / (committed + p + eps)` (dilution), 2 = exclusive
`p / (committed + eps)` (temporal attention's denominator, S18). Only `inv` and
the backward's direct term differ between 1 and 2; the suffix term
`-p * inv^2` is common. Both the forward and the backward must be launched with
the same code -- the backward previously used `int(bool(share_cumsum))`, which
silently computed the inclusive gradient for mode 2 (fixed; the parity test
caught it before any run).

## Precision and limits

- `dot_precision`: "ieee" for parity tests, "tf32" for fp32 activations,
  "bf16" (default) for bf16 activations. fp32 accumulators and fp32
  cumsum/share math in every mode.
- Tiles 64x64, 4 warps everywhere, except BA at 32 rows x 64 keys, KF at 32
  rows (64 from 4K tokens) x 64 keys, and the forward-only path at 128 rows x
  32 keys; forward num_stages=2, backward 1 except BA, which runs 2 stages
  with bf16 operands. fp32 operand tiles keep one stage, since two can exceed
  shared memory at larger head dims. An 88-combination sweep of {64,128}^2 x
  {4,8} warps x {1,2,3} stages (forward) and {64,128}^2 x {4,8}^2 x {1,2}
  (backward) found nothing better for the training kernels: every 128-wide
  tile is out of shared memory or slower, and every 8-warp forward is slower.
- Every launch names every kernel argument, defaults included. Under
  `torch.compile`, which training uses, the launches go through PyTorch's
  Triton wrapper, which compiles a kernel from the arguments it is given and
  fails on one left to its default. tests/test_kernels.py compiles each
  forward-only and training path and compares it with its eager run.
- The forward-only path's fp32 operands are supported up to head_dim 128.
- head_dim >= 16 required (Triton dot minimum); the model falls back to the
  tensor path below that.
- IEEE fp32 D128 can exceed shared memory with the default two-stage
  forward on this GPU; `dilution_attention_forward(..., num_stages=1)`
  works for the saved-correction parity test.
- Training retains an extra fp32 `(B, H, T, D)` tensor per dilution layer
  for pk: 64 MiB at B8 H8 T4096 D64, or 32 MiB at B2 H8 T8192 D64.
- The saved carry checkpoints come every 32 rows on the training path, so
  that buffer is `(B*H, T/32, T)` fp32: 128 MiB per layer at B2 H8 T8192,
  256 MiB at B1 H8 T16384; +0.75 GiB peak at the 12-layer 8K training config
  (10.54 -> 11.29 GiB) and about 1.9 GiB at 16K.
- Training and the key-span forward use atomics, so they match the reference
  to tolerance but are not bitwise deterministic. The row-parallel forward and
  both decode forms are deterministic.

## Measured

Forward plus backward of the attention op inside the model step (torch
profiler, bf16, D64, RTX PRO 6000 Blackwell), before and after the slimmed tile
bodies, ms:

    shape             before   after     K1           KF           BA            BC
    B32 H12 T1024      3.68     3.67    0.23 0.21    0.69 0.68    1.89  1.93    0.22 0.20
    B8  H12 T2048      2.12     1.83    0.21 0.19    0.47 0.44    1.03  0.83    0.20 0.17
    B8  H8  T4096      5.42     4.57    0.51 0.45    1.29 1.09    2.75  2.23    0.51 0.47
    B2  H8  T8192      5.05     4.19
    B2  H12 T16384    30.5     25.9     2.97 2.65    7.57 6.58   16.3  13.4     3.01 2.69

The remainder at B8 H8 T4096 is normalize 0.03, the dq epilogue 0.09 and torch
glue 0.22 ms. The gain grows with length, 14-16% from 2K to 16K, and is nil at
1K: programs there run about 17 row blocks each and their fixed cost dominates.

Model throughput (median tok/s over steps 200-1000 of a screen): 159.9K for the
64M model at 8K against about 280K for SDPA softmax at the same shape, and 33.4K
for the 209M dilution model at 16K against 72-75K for softmax. From the first
working kernels, model throughput rose 4% / 19% / 30% / 41% / 52% at 1K / 2K /
4K / 8K / 16K, and the last pass added 2.6-4.1%, with loss unchanged; a
full-schedule replay reproduces the first kernels' loss to 0.0007 nats (S26).
These throughputs predate the slimmed tile bodies.

Prefill (`return_aux="totals"`, what `begin_cache` runs) at B1 H12 D64 bf16,
time and peak allocated memory (`scripts/bench_prefill_decode.py`): the
row-parallel path with the full carry, the key-span path as first written
(64 x 64 tiles, masked bodies), and now. The softmax column is causal
`scaled_dot_product_attention`:

    T        softmax     row-parallel            key spans, first       now                    now / softmax
    16K      1.35 ms     4.97 ms  /  0.42 GiB    4.44 ms / 0.05 GiB     3.19 ms / 0.05 GiB     2.4x
    64K      19.3 ms     86.5 ms  /  6.20 GiB    70.0 ms / 0.20 GiB     48.2 ms / 0.20 GiB     2.5x
    128K     78.0 ms     368 ms   / 24.4 GiB     276 ms  / 0.40 GiB     191 ms  / 0.40 GiB     2.4x
    256K     315 ms      3100 ms  / 96.8 GiB     1133 ms / 0.80 GiB     771 ms  / 0.80 GiB     2.4x
    512K     1.31 s      did not fit             4.67 s  / 1.59 GiB     3.13 s  / 1.59 GiB     2.4x
    1M       5.61 s      did not fit             20.4 s  / 3.19 GiB     13.6 s  / 3.19 GiB     2.4x

The plain forward (`return_aux=False`) takes the same path and the same time
as prefill. Short calls: key-span over row-parallel time is 0.77 / 0.68 / 0.61 /
0.57 at 4K / 8K / 16K / 32K (B1 H12), 0.70 at B8 H12 T4096 and 0.81 with a single
head at 4K; below 4K it ranges from 0.8 to 1.45 depending on the number of
rows, so the threshold stays at 4096 tokens.

Profiling tools: `scripts/step_profile.py` times each kernel in the order the
step runs it (isolated timing can misreport in-sequence cost by up to 1.5 ms),
`scripts/profile_kernels.py` times kernels in isolation,
`scripts/bench_crossover.py` compares the two training forward paths at short
context, and `scripts/bench_prefill_decode.py` times prefill, the plain forward
and a decode step against softmax at long context, with peak memory.

## Where the time goes

BA is 52% of the step (2.82 of 5.46 ms at B8 H8 T4096; 16.2 of 30.2 ms at the
209M 16K shape). Its SASS runs about 1,650 instructions per warp per 32x64 tile,
including 80 tensor-core ops, 29 barriers, about 220 shared-memory accesses and
23 spill accesses. With two blocks of 4 warps per SM, it issues at about 27% of
the SM's rate: latency, not arithmetic, bounds it. Removing one piece at a time
(timing only): atomics 0.33 ms, the two row scans 0.33, the dv product 0.29,
reciprocals 0.08. No single piece dominates.

Triton's scan cannot run in the tensor-core accumulator layout, so each of BA's
two scans converts the tile through shared memory to a row-split layout and
back (4 of the loop's 6 layout conversions). Gluon in Triton 3.7.1 cannot lower
a scan in that layout or its linear equivalent either. An inline PTX
warp-shuffle scan on the accumulator layout is exact (1.5e-7 relative) and
removes those conversions (BA barriers 45 -> 33, spills 32 -> 16), but saves
nothing on top of the prefetch and 4% of KF. It is not adopted: its correctness
depends on the compiler's choice of accumulator layout, which the language
cannot assert.

Structurally, p is rebuilt four times per tile per step, about 4.9
matmul-equivalents against a theoretical 4, and the cumsum chain (the in-tile
scans, the share division and its backward) is about a quarter of the step
(S4). Launch configuration, tile shapes, pass ordering, atomics placement and
the other levers reachable from Triton have been measured (S26); further gains
need register-level scheduling of BA that Triton does not expose, or a
different decomposition.

### Instruction counts

Per warp and per tile, the loops are mostly scalar bookkeeping around a few
tensor-core products. Before the slimmed bodies, per tile (HMMA are the
tensor-core instructions):

    kernel (tile)            instructions   HMMA   after (unmasked body)
    K1 + pk (64 x 64)              542        64        365
    KF (32 x 64)                   858        32        562
    BA (32 x 64)                 1,660        80      1,015
    BC (64 x 64)                   484        64        323
    share sweep (64 x 64)        1,700        64        845 at 128 x 32

Counting a tensor-core instruction as 8.4 scalar ones (fitted on the softmax
kernel and the share sweep) predicted the four training kernels' times to
within 25% before the change. Mask logic was about a fifth of every loop
(index compares, predicate packing, three to five selects per element), the
full-range division 8-10 instructions per element, and each column reduction
in the tensor-core layout about 150.

The model overstates what removing instructions buys where something else
binds. K1 and BC gained 10% and 8% for a third fewer instructions; BA gained
17% for 39% fewer; KF gained 4% for 35% fewer. Removing one piece at a time
from the unmasked KF at B2 H12 T16384 (6.63 ms, timing only): the in-tile scan
1.7 ms, the output atomics 1.15, the value product with them 2.0, the checkpoint
store 0.17; without all of them 2.3 ms remain. The scan is the largest piece: it
leaves the tensor-core layout through shared memory and comes back, with 22
barriers per tile at 32 rows and 17 at 64, which is why taller KF tiles help.

Tried on the slimmed bodies and not adopted:

- Folding the row normaliser into the exponent (`exp2(aff - (rmax + log2
  rden))`) and eps into the carry: 3% of the share sweep together, and it
  changes how p is rounded in every kernel.
- `div.approx` and the square of `rsqrt` in place of `rcp.approx`: the same
  speed; the latter is portable Triton but 1.5 ulp.
- Scanning the transposed tile, so that the scan axis is the layout's fast
  axis: Triton picks a different blocked layout, with the same number of
  barriers and the same time.
- The in-tile prefix sums as a bf16 tensor-core product with a triangular ones
  matrix: 19% faster for KF at 32 rows (6.62 -> 5.36 ms), slower at 64 rows
  (5.72 against 5.50), and it moves outputs by 4.3e-3 against 2.6e-4 for the
  existing bf16 rounding. Rejected for precision.
- BA at 64-row tiles (24.1 against 13.6 ms), BA and BC on 32-key tiles (BA
  20.0), BA at 8 warps (21.1).
- Other tiles for the training K1 and BC: 128 x 64 saves 0.2 ms of K1's 2.6 at
  252 registers, 64 x 128 saves 0.2 ms of BC's 2.7 with 30 spills; under 1% of
  a step each.
- Single-warp programs: slower everywhere, and two shapes (16 x 256 and
  32 x 256 with one warp, 500-2,000 spills) returned wrong outputs.

### Prefill

Two sweeps is the floor without quadratic storage: p needs the completed row
statistics, and the column sums need p. At 64K the statistics sweep is 11 ms
and the share sweep 37 ms, against 19 ms for the whole softmax kernel, which is
what keeps prefill at 2.4-2.5x softmax. On the first key-span kernel (64 x 64
tiles, masked body, 54 ms), removing one piece at a time (timing only, span 8):
the value product 12.1 ms, the in-tile scan 10.8, the column-sum load, store
and reduction 7.8, the division 5.1, the causal masks 4.9, the exponential 1.3;
with all of those gone the tile loads and loop still cost 12.7 ms. The masks,
the division and the column reduction are what the slimmed body removed; the
scan and the value product remain.

The exponential is not the bottleneck here, so computing part of it as a
polynomial on the FMA units (the FlashAttention-4 trick) was not attempted: it
could recover at most 1.3 of 48 ms.

Before the slimming the tile shape hardly mattered (64K: statistics sweep
15.7 ms at 64 x 64 against 12.7 at 128 x 32; share sweep 54.1 against 50.0;
8 warps 110 ms; 128 x 128 out of shared memory). The share sweep compiles to
255 registers, but occupancy is not the limit: capping it (`maxnreg` 200 / 168 /
128) gives 62 / 76 / 107 ms, and the 64 x 32 tile, which fits 168 registers
with 6 spills, is no faster (57.9 against 56.2 ms).

Giving the *training* forward key spans, to cut its output atomics the same
way, is slower: at 32-row tiles with the carry checkpoints still stored, spans
of 1 / 2 / 4 / 8 tiles take 1.49 / 1.45 / 1.45 / 1.58 ms against KF's 1.16 at
B8 H8 T4096, and 9.42 / 8.79 / 8.14 / 7.87 against 7.31 ms at B2 H12 T16384.
Only B32 H12 T1024 at span 4 gains (0.63 against 0.69 ms). Re-reading the key
and value tiles and the sums costs more than the atomics it saves; KF is
unchanged.

Open: the training path still saves a `(N, T/32, T)` carry for the backward
(25 GiB per layer per sequence at 128K), so training past about 32K needs the
query-chunk idea or recomputation in the backward as well.
