#!/bin/bash
# Broader evaluation of the Kimi pair, public tooling only, at the settings of the published DilutionAttention /
# Selective-Attention-Study results so the rows are comparable. Run after runs/run_kimi.sh:
#   SEEDS="1337 2357 7331" PG19=<prepared PG-19 dir> bash runs/eval_kimi_extra.sh
#   1. PG-19 perplexity by position to 131,072 (scripts/long_perplexity.py; published results/longppl/pg19_209m_16k_1x.json).
#   2. Distant-text control: full prefix vs a 16K rolling window to 131,072 (scripts/investigate/rolling_window.py, from
#      Selective-Attention-Study 7e7d591; released results/investigate/rolling_window_16k_1x.json).
#   3. Niah hard mode (natural PG-19 filler; 0 and 3 keyed distractors) at 4K-128K (scripts/investigate/niah_hard.py, from
#      Selective-Attention-Study; released niah_hard_d0.json / niah_hard_d3.json).
#   4. Attention onto the needle per layer at 7,936 tokens (scripts/investigate/needle_attention_hybrid.py; published
#      results/needle/*_ctx7936.json) and attention structure of the six attention layers at 1,024 / 4,096 tokens.
#   5. The standard NIAH ladder on the kept 16,000 / 32,000 checkpoints (16K, 64K, 128K): is retrieval stable in training?
SEEDS=${SEEDS:-1337}
cd "$(dirname "$0")/.."
export PYTHONPATH=src DILUTION_NIAH_ROOT=${DILUTION_NIAH_ROOT:-$(cd .. && pwd)/Niah}
PG19=${PG19:-data/pg19_gpt2}                          # scripts/long_perplexity.py prepare (emozilla/pg19, pinned)
FILLER=data/niah_filler/pg19_train6.txt               # scripts/investigate/make_niah_filler.py --books 6
mkdir -p results/longppl results/investigate results/needle results/structure results/niah results/queue
log() { echo "[kimi_extra] $(date -Is) $*"; }
ATTN_LAYERS="3 7 11 15 19 23"                         # [KDA, KDA, KDA, attention] x 6, 0-indexed
for SEED in $SEEDS; do
  models=()
  for ARM in kimi_dilution kimi_softmax_nope; do
    ck=runs/scale_16384/${ARM}_seed$SEED/checkpoint_step_0064000.pt
    [ -f $ck ] && models+=(--model ${ARM}_s$SEED=$ck) || log "missing $ck"
  done
  [ ${#models[@]} -gt 0 ] || continue
  log "seed $SEED: PG-19 perplexity by position"
  [ -f results/longppl/pg19_kimi_s$SEED.json ] || python scripts/long_perplexity.py eval --data $PG19 --max-length 131072 \
    "${models[@]}" --output results/longppl/pg19_kimi_s$SEED.json >> results/queue/longppl.txt 2>&1 || log "longppl failed"
  log "seed $SEED: rolling-window control"
  [ -f results/investigate/rolling_window_kimi_s$SEED.json ] || python scripts/investigate/rolling_window.py --data $PG19 \
    --max-length 131072 "${models[@]}" --output results/investigate/rolling_window_kimi_s$SEED.json \
    >> results/queue/rolling.txt 2>&1 || log "rolling window failed"
  for D in 0 3; do
    log "seed $SEED: NIAH hard mode, $D distractors"
    python scripts/investigate/niah_hard.py --filler $FILLER --lengths 4096 16000 32000 64000 128000 --samples 10 \
      --distractors $D "${models[@]}" --output results/investigate/niah_hard_d${D}_kimi_s$SEED.json \
      >> results/queue/niah_hard.txt 2>&1 || log "niah hard d$D failed"
  done
  for ARM in kimi_dilution kimi_softmax_nope; do
    run=runs/scale_16384/${ARM}_seed$SEED
    [ -f $run/latest.pt ] || continue
    log "seed $SEED: $ARM needle attention and structure"
    [ -f results/needle/${ARM}_seed${SEED}_ctx7936.json ] || python scripts/investigate/needle_attention_hybrid.py $run \
      --context 7936 --output results/needle/${ARM}_seed${SEED}_ctx7936.json >> results/queue/needle.txt 2>&1 \
      || log "needle attention failed $ARM"
    for S in 1024 4096; do
      [ -f results/structure/${ARM}_seed${SEED}_seq$S.json ] || python scripts/attention_structure.py $run --seq $S \
        --layers $ATTN_LAYERS --output results/structure/${ARM}_seed${SEED}_seq$S.json >> results/queue/structure.txt 2>&1 \
        || log "structure failed $ARM seq $S"
    done
    for C in 16000 32000; do
      k=$run/checkpoint_step_00$C.pt; out=results/niah/kept${C}__${ARM}_seed$SEED.json
      [ -f $k ] && [ ! -f $out ] || continue
      log "seed $SEED: $ARM NIAH at step $C"
      python -m dilution.niah_eval --checkpoint $k --context-length 16000 64000 128000 --context-override 132000 \
        --samples 10 --num-needles 8 --output $out >> results/queue/niah.txt 2>&1 || log "niah failed $ARM step $C"
    done
  done
done
log "KIMI_EXTRA_DONE seeds $SEEDS"
