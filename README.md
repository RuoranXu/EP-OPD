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
bash run_ep_opd.sh
```

### Evaluation

```bash
bash test/run.sh /path/to/baseline_checkpoint /path/to/ep_opd_checkpoint
```
