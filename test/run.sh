#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if (( $# < 2 || $# > 3 )); then
  printf 'Usage: bash test/run.sh BASELINE_MODEL EP_OPD_MODEL [OUTPUT_DIR]\n' >&2
  exit 2
fi
baseline_model="$1"
ep_opd_model="$2"
eval_dir="${3:-${EVAL_DIR:-outputs/evaluation}}"
python_bin="${PYTHON:-python}"
api_key="${DEEPSEEK_API_KEY:-}"
unset DEEPSEEK_API_KEY
trap 'unset DEEPSEEK_API_KEY api_key' EXIT
mkdir -p -- "$eval_dir"

for method in baseline ep_opd; do
  if [[ "$method" = baseline ]]; then model="$baseline_model"; else model="$ep_opd_model"; fi
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "$python_bin" -m test.generate \
    --model "$model" --split test --max-new-tokens 32768 \
    --output "$eval_dir/${method}_test.jsonl"
  "$python_bin" -m test.evaluate \
    --predictions "$eval_dir/${method}_test.jsonl" --split test
done

if [[ -z "$api_key" ]]; then
  read -r -s -p "DeepSeek API key: " api_key
  printf '\n'
fi
if [[ -z "$api_key" ]]; then
  printf 'DeepSeek API key is required.\n' >&2
  exit 2
fi
export DEEPSEEK_API_KEY="$api_key"
unset api_key
for method in baseline ep_opd; do
  "$python_bin" -m test.judge_csp \
    --predictions "$eval_dir/${method}_test.jsonl" \
    --split test --workers 6 \
    --legacy-directory "$eval_dir/no_legacy_labels"
done
unset DEEPSEEK_API_KEY

"$python_bin" -m test.compare \
  --baseline "$eval_dir/baseline_test.jsonl" \
  --candidate "$eval_dir/ep_opd_test.jsonl" \
  --replicates 10000 --output "$eval_dir/comparison.json"

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "$python_bin" -m test.generate \
  --model "$ep_opd_model" --qasper --max-new-tokens 32768 \
  --output "$eval_dir/ep_opd_qasper.jsonl"
"$python_bin" -m test.evaluate \
  --predictions "$eval_dir/ep_opd_qasper.jsonl" --qasper
