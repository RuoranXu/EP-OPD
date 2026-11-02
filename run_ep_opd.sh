#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python_bin="${PYTHON:-python}"
model_size="${MODEL_SIZE:-4b}"
if [[ -z "${INIT_EPOCHS:-}" ]]; then
  read -r -p "EP-OPD initialization epochs (positive integer): " INIT_EPOCHS
fi
if [[ -z "${EP_EPOCHS:-}" ]]; then
  read -r -p "EP-OPD epochs (positive integer): " EP_EPOCHS
fi
if [[ ! "$INIT_EPOCHS" =~ ^[1-9][0-9]*$ || ! "$EP_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Epoch counts must be positive integers.\n' >&2
  exit 2
fi

"$python_bin" main.py method=sft trainer.size="$model_size" \
  "+sft_overrides.sft.epochs=$INIT_EPOCHS"
export SFT_MODEL_PATH="${CHECKPOINT_DIR:-outputs/checkpoints}/sft_${model_size}_seed42/final"
"$python_bin" main.py method=ep_opd trainer.size="$model_size" \
  "+backend_overrides.training.epochs=$EP_EPOCHS"
