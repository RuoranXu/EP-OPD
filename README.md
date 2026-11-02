# EP-OPD

## Quick Start

EP-OPD uses VERL/Ray with four CUDA GPUs on one Linux node. Use Python 3.10.

### Setup

```bash
git clone https://github.com/RuoranXu/EP-OPD.git
cd EP-OPD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_SIZE=4b
export STUDENT_MODEL_PATH=/absolute/path/to/Qwen3-4B-Base
export TEACHER_MODEL_PATH=/absolute/path/to/Qwen3-30B-A3B-Instruct-2507
export CHECKPOINT_DIR="$PWD/outputs/checkpoints"
```

Set the model paths to local directories containing the model weights and tokenizer files.
The bundled data is prepared automatically under `data/cache/` on the first run.
Epoch counts have no defaults; enter your own positive integer for each stage.

### EP-OPD

```bash
read -r -p "EP-OPD initialization epochs (positive integer): " INIT_EPOCHS
read -r -p "EP-OPD epochs (positive integer): " EP_EPOCHS
python main.py method=sft trainer.size="$MODEL_SIZE" \
  "+sft_overrides.sft.epochs=${INIT_EPOCHS:?Set INIT_EPOCHS}" &&
export SFT_MODEL_PATH="$CHECKPOINT_DIR/sft_${MODEL_SIZE}_seed42/final" &&
python main.py method=ep_opd trainer.size="$MODEL_SIZE" \
  "+backend_overrides.training.epochs=${EP_EPOCHS:?Set EP_EPOCHS}"
```

### Evaluation of paper metrics

```bash
export BASELINE_MODEL=/absolute/path/to/baseline_checkpoint
export EP_OPD_MODEL=/absolute/path/to/ep_opd_checkpoint
export EVAL_DIR="$PWD/outputs/evaluation"
mkdir -p "$EVAL_DIR"

for method in baseline ep_opd; do
  if [ "$method" = baseline ]; then model="$BASELINE_MODEL"; else model="$EP_OPD_MODEL"; fi
  CUDA_VISIBLE_DEVICES=0 python -m test.generate \
    --model "$model" --split test --max-new-tokens 32768 \
    --output "$EVAL_DIR/${method}_test.jsonl"
  python -m test.evaluate \
    --predictions "$EVAL_DIR/${method}_test.jsonl" --split test
done

read -r -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_API_KEY

for method in baseline ep_opd; do
  python -m test.judge_csp \
    --predictions "$EVAL_DIR/${method}_test.jsonl" \
    --split test --workers 6 \
    --legacy-directory "$EVAL_DIR/no_legacy_labels"
done

unset DEEPSEEK_API_KEY

python -m test.compare \
  --baseline "$EVAL_DIR/baseline_test.jsonl" \
  --candidate "$EVAL_DIR/ep_opd_test.jsonl" \
  --replicates 10000 --output "$EVAL_DIR/comparison.json"

CUDA_VISIBLE_DEVICES=0 python -m test.generate \
  --model "$EP_OPD_MODEL" --qasper --max-new-tokens 32768 \
  --output "$EVAL_DIR/ep_opd_qasper.jsonl"
python -m test.evaluate \
  --predictions "$EVAL_DIR/ep_opd_qasper.jsonl" --qasper
```
