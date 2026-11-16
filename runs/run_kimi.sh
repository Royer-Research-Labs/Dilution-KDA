#!/bin/bash
# Kimi Linear's layer pattern at 209M / 16K in the public DilutionAttention harness: the attention slots are dilution
# (kimi_dilution) or no-position softmax (kimi_softmax_nope, the reference). Activate an environment with the harness
# dependencies and fla-core first; set DILUTION_NIAH_ROOT to a Niah checkout (default ../Niah).
#   SEEDS="1337 2357 7331" bash runs/run_kimi.sh
#   1. GPU tests (the public kernel tests and fla's KDA kernel against the reference); stop on failure.
#   2. Per arm: train on the published schedule to step 64,000 (kept 16,000 / 32,000), the common-slice loss (published
#      script, 64 x 2 windows), and the published NIAH ladder (256-16,000 at 20 placements; 16K-64K, 64K-128K and, for
#      dilution, 192K-256K at 10). Interrupted training resumes from latest.pt; finished steps are skipped.
SEEDS=${SEEDS:-1337}
cd "$(dirname "$0")/.."
export PYTHONPATH=src DILUTION_NIAH_ROOT=${DILUTION_NIAH_ROOT:-$(cd .. && pwd)/Niah}
mkdir -p results/eval results/niah results/queue runs/scale_16384
log() { echo "[kimi] $(date -Is) $*"; }
L16="256 512 896 1024 1920 3072 3840 6144 7936 12288 16000"
STOP=64000

python -m pytest -q tests > results/queue/pytest_gpu.txt 2>&1
tail -1 results/queue/pytest_gpu.txt | while read -r l; do log "pytest: $l"; done
tail -1 results/queue/pytest_gpu.txt | grep -qE "failed|error" && { log "GPU TESTS FAILED: stopping"; exit 1; }

for SEED in $SEEDS; do
  for ARM in kimi_dilution kimi_softmax_nope; do
    run=runs/scale_16384/${ARM}_seed$SEED; name=scale1x__${ARM}_seed$SEED
    if ! grep -q '"event": "stopped"' $run/metrics.jsonl 2>/dev/null; then
      resume=""; [ -f $run/latest.pt ] && resume="--resume $run/latest.pt"
      log "train $ARM seed $SEED to step $STOP"
      python -m dilution.train --config configs/scale_16384/$ARM.yaml --seed $SEED --output-dir $run $resume \
        --stop-at-step $STOP --keep-checkpoint-at 16000 32000 >> results/queue/train_${ARM}_seed$SEED.txt 2>&1 \
        || { log "TRAIN FAILED $ARM seed $SEED"; continue; }
    fi
    [ -f results/eval/$name.json ] || python scripts/eval_checkpoint.py $run --batches 64 --batch-size 2 \
      --output results/eval/$name.json >> results/queue/eval.txt 2>&1 || log "eval failed $name"
    [ -f results/niah/$name.json ] || python -m dilution.niah_eval --checkpoint $run/latest.pt --context-length $L16 \
      --samples 20 --num-needles 8 --output results/niah/$name.json >> results/queue/niah.txt 2>&1 || log "niah failed $name"
    specs="extrap64k:66000:16000 24000 32000 48000 64000|extrap128k:132000:64000 96000 128000"
    [ $ARM = kimi_dilution ] && specs="$specs|extrap256k:260000:192000 256000"
    IFS="|" read -ra parts <<< "$specs"
    for p in "${parts[@]}"; do
      tag=${p%%:*}; rest=${p#*:}; override=${rest%%:*}; lengths=${rest#*:}
      out=results/niah/${tag}__${ARM}_seed$SEED.json
      [ -f $out ] || python -m dilution.niah_eval --checkpoint $run/latest.pt --context-length $lengths \
        --context-override $override --samples 10 --num-needles 8 --output $out >> results/queue/niah.txt 2>&1 \
        || log "niah failed $tag $ARM"
    done
    log "done $ARM seed $SEED: $(python - $name <<'PY'
import json, sys
n = sys.argv[1]
try:
    loss = json.load(open(f"results/eval/{n}.json"))["loss"]
    print(f"loss {loss:.4f}")
except Exception as e:
    print(f"? ({e})")
PY
)"
  done
done
log "KIMI_DONE seeds $SEEDS"
