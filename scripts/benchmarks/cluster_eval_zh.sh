#!/usr/bin/env bash
# Development evaluation of one zh checkpoint on a GPU node (public sets only; blind and test are never read).
#   cluster_eval_zh.sh <checkpoint.pt> <name> [task|lm]
# Writes outputs/eval-zh/<name>/:
#   pinyin/      ime-pinyin-zh-v1 development: Vime search (beam 16 per key position, in-field context),
#                top 10 rescored by vimeml-tiny-zh-v1 after the same context
#   standard/    ime-standard-zh-v1 development full pinyin rerank, compared with vimeml-tiny-zh-v1
#   predict/     ime-predict-zh-v1 development, all tracks
#   report.md    pinyin table and paired tests against the references below
# References (computed once, then reused from outputs/eval-zh/ref-*): v1 in lm mode and r2-cont in task
# mode on the same pinyin run, and v1's Standard scores. Needs, besides the repo: the 12K tokenizer,
# the frozen benchmark directories (pinyin: development.jsonl, development-contexts.jsonl and
# manifest.json only), Vime's readings dictionary and the v1 / r2-cont checkpoints.
# A failed step is logged and the rest still run.
# Parallel jobs: VIMEML_REFS=0 skips the reference pinyin runs. Naming a job ref-v1-lm (v1, lm) or
# ref-r2-cont (r2-cont, task) writes exactly the reference directories, so three jobs can run at once;
# rerun only the report line at the end when they are all done.
set -uo pipefail
cd "$(dirname "$0")/../.."
ckpt="${1:?usage: cluster_eval_zh.sh <checkpoint.pt> <name> [task|lm]}"
name="${2:?name}"
mode="${3:-task}"
[[ -f "$ckpt" ]] || { echo "No checkpoint $ckpt" >&2; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
unset PYTHONPATH
py="${VIMEML_PYTHON:-python3}"
device="${VIMEML_DEVICE:-cuda}"
v1="artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt"
r2="${VIMEML_R2_CONT:-artifacts/models/tiny-zh13-12k-pinyin-r2-cont/final.pt}"
tok="artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"
std="outputs/benchmarks/ime-standard-zh-v1/frozen-r1"
out="outputs/eval-zh/$name"
mkdir -p "$out" outputs/eval-zh
step() { echo "== $(date '+%F %T') $*"; }
pinyin() {  # <checkpoint> <mode> <dir>
  [[ -f "$3/rows.jsonl" ]] && { echo "reuse $3"; return 0; }
  "$py" -X utf8 scripts/benchmarks/pinyin_gen_zh.py run --checkpoint "$1" --mode "$2" --out "$3" --device "$device"
}
standard() {  # <checkpoint> <dir>
  [[ -f "$2/scores.jsonl" ]] || "$py" -X utf8 scripts/benchmarks/score_zh13_lm.py "$1" "$tok" "$2/scores.jsonl" --device "$device" || return 1
  [[ -d "$2/full" ]] || "$py" -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py score "$std" "$2/full" --track full --scores "$2/scores.jsonl"
}

if [[ "${VIMEML_REFS:-1}" != 0 ]]; then
  step "pinyin references"
  pinyin "$v1" lm outputs/eval-zh/ref-v1-lm/pinyin || echo "FAIL ref v1 pinyin"
  if [[ -f "$r2" ]]; then pinyin "$r2" task outputs/eval-zh/ref-r2-cont/pinyin || echo "FAIL ref r2-cont pinyin"; fi
fi
step "pinyin $name ($mode)"
pinyin "$ckpt" "$mode" "$out/pinyin" || echo "FAIL pinyin"

step "Standard"
mkdir -p outputs/eval-zh/ref-v1-lm/standard "$out/standard"
standard "$v1" outputs/eval-zh/ref-v1-lm/standard || echo "FAIL ref Standard"
standard "$ckpt" "$out/standard" || echo "FAIL Standard"
"$py" -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py compare outputs/eval-zh/ref-v1-lm/standard/full \
  "$out/standard/full" "$out/standard/vs-v1.json" || echo "FAIL Standard compare"

step "predict"
"$py" -X utf8 scripts/benchmarks/predict_ime_zh.py run --model vime --checkpoint "$ckpt" --tokenizer "$tok" \
  --device "$device" --out "$out/predict" || echo "FAIL predict"
"$py" -X utf8 scripts/benchmarks/predict_ime_zh.py score --run "$out/predict" || echo "FAIL predict score"

step "report"
runs=("v1-lm=outputs/eval-zh/ref-v1-lm/pinyin")
[[ -f outputs/eval-zh/ref-r2-cont/pinyin/rows.jsonl ]] && runs+=("r2-cont=outputs/eval-zh/ref-r2-cont/pinyin")
runs+=("$name=$out/pinyin")
"$py" -X utf8 scripts/benchmarks/pinyin_gen_zh.py report "${runs[@]}" --base v1-lm --out "$out/report.md" || echo "FAIL report"
step done
