#!/usr/bin/env bash
# Print run_gpu_queue.sh job lines for the development evaluation of one or more zh checkpoints, split so
# that the parts run on separate GPUs at once (the main part of cluster_eval_zh.sh, as a queue).
#   jobs_eval_zh.sh NAME=CHECKPOINT[:MODE] ... > jobs.txt && scripts/run_gpu_queue.sh jobs.txt
# MODE task (default) or lm. Environment: SHARDS (pinyin generation shards per checkpoint, default 4),
# PARTS (any of pinyin standard predict typo; default all four). Each checkpoint writes outputs/eval-zh/NAME/:
#   pinyin/       generation, beam 16 per key position, in-field context, prior = the checkpoint itself
#   standard/     ime-standard-zh-v1 development full pinyin, compared with outputs/eval-zh/ref-v1-lm/standard
#   predict/      ime-predict-zh-v1 development, all tracks
#   typo/         ime-typo-zh-v1 lm-rule scores, then calibration.json (lambda 1.6, prior self)
#   report.md     pinyin table against outputs/eval-zh/ref-v1-lm/pinyin
# Every independent part comes first; one "wait", then the shard merges and the reports.
set -euo pipefail
(( $# )) || { echo "usage: jobs_eval_zh.sh NAME=CHECKPOINT[:MODE] ..." >&2; exit 2; }
shards="${SHARDS:-4}"
parts="${PARTS:-pinyin standard predict typo}"
py='"${VIMEML_PYTHON:-python3}" -X utf8'
tok=artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model
std=outputs/benchmarks/ime-standard-zh-v1/frozen-r1
after=()
for spec in "$@"; do
  name="${spec%%=*}"; rest="${spec#*=}"
  ckpt="${rest%%:*}"; mode=task; [[ "$rest" == *:* ]] && mode="${rest##*:}"
  out="outputs/eval-zh/$name"
  for part in $parts; do
    case "$part" in
      pinyin)
        list=""
        for ((i = 0; i < shards; i++)); do
          echo "$py scripts/benchmarks/pinyin_gen_zh.py run --checkpoint $ckpt --prior $ckpt --mode $mode --device cuda --shard $i/$shards --out $out/pinyin-shard-$i"
          list+=" $out/pinyin-shard-$i"
        done
        after+=("$py scripts/benchmarks/pinyin_gen_zh.py merge$list --out $out/pinyin && $py scripts/benchmarks/pinyin_gen_zh.py report v1-lm=outputs/eval-zh/ref-v1-lm/pinyin $name=$out/pinyin --base v1-lm --out $out/report.md") ;;
      standard)
        echo "mkdir -p $out/standard && $py scripts/benchmarks/score_zh13_lm.py $ckpt $tok $out/standard/scores.jsonl --device cuda && $py -I scripts/benchmarks/score_standard_ime_zh.py score $std $out/standard/full --track full --scores $out/standard/scores.jsonl && $py -I scripts/benchmarks/score_standard_ime_zh.py compare outputs/eval-zh/ref-v1-lm/standard/full $out/standard/full $out/standard/vs-v1.json" ;;
      predict)
        echo "$py scripts/benchmarks/predict_ime_zh.py run --model vime --checkpoint $ckpt --tokenizer $tok --device cuda --out $out/predict && $py scripts/benchmarks/predict_ime_zh.py score --run $out/predict" ;;
      typo)
        echo "$py scripts/benchmarks/typo_zh.py score --checkpoint $ckpt --rules lm --device cuda --out $out/typo && $py scripts/benchmarks/typo_zh.py calibration --rows $out/typo --lambda 1.6 --prior self --out $out/calibration.json" ;;
      *) echo "Unknown part $part" >&2; exit 2 ;;
    esac
  done
done
if (( ${#after[@]} )); then
  echo wait
  printf '%s\n' "${after[@]}"
fi
