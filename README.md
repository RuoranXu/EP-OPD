# EP-OPD

## Quick Start

Run SFT first, then start EP-OPD in a separate command after SFT finishes.
Both stages use VERL/Ray with four CUDA GPUs on one Linux node. Use Python 3.10.

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

### 1. SFT

```bash
read -r -p "SFT epochs (positive integer): " SFT_EPOCHS
python main.py method=sft trainer.size="$MODEL_SIZE" \
  "+sft_overrides.sft.epochs=${SFT_EPOCHS:?Set SFT_EPOCHS}"
```

Wait for this command to finish successfully. The final SFT weights are saved to:

```text
$CHECKPOINT_DIR/sft_${MODEL_SIZE}_seed42/final
```

Use a fresh `CHECKPOINT_DIR` when starting a new SFT run.

### 2. EP-OPD

Run this separately in the same shell, keeping the environment variables from setup:

```bash
export SFT_MODEL_PATH="$CHECKPOINT_DIR/sft_${MODEL_SIZE}_seed42/final"
test -f "$SFT_MODEL_PATH/config.json"
read -r -p "EP-OPD epochs (positive integer): " EP_EPOCHS
python main.py method=ep_opd trainer.size="$MODEL_SIZE" \
  "+backend_overrides.training.epochs=${EP_EPOCHS:?Set EP_EPOCHS}"
```

EP-OPD starts from the SFT weights above. Its checkpoints and training logs are written to
`$CHECKPOINT_DIR/ep_opd_${MODEL_SIZE}_seed42/`.
Evaluation is run separately using the scripts in `test/`.

For the 1.7B student, set `MODEL_SIZE=1p7b` and point `STUDENT_MODEL_PATH` to
`Qwen3-1.7B-Base`, then run the same two stages in order.
