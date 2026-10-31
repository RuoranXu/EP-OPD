import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def sft_identity(config):
    return dict(base_student=str(Path(config['models']['base_student']).resolve()),
        cache_sha256=sha256(config['data']['cache']), settings=config['sft'], seed=42,
        trainer_sha256=sha256(ROOT / 'trainer/sft.py'),
        optimizer={key: config['training'].get(key) for key in
                   ['betas', 'weight_decay', 'fused_adamw', 'optimizer_precision', 'gradient_checkpointing']},
        attention_backend=config['teacher'].get('attention_backend', 'sdpa'),
        trust_remote_code=config['models'].get('trust_remote_code', False))


def verify_sft(checkpoint, identity):
    checkpoint = Path(checkpoint)
    folder = checkpoint.parent
    try:
        complete = json.loads((folder / 'training_complete.json').read_text(encoding='utf8'))
        manifest = json.loads((folder / 'run_manifest.json').read_text(encoding='utf8'))
    except (OSError, ValueError) as error:
        raise ValueError('SFT is incomplete; use a completed SFT checkpoint or a fresh CHECKPOINT_DIR') from error
    if (complete.get('smoke', True) or not complete.get('maximum_updates')
            or complete.get('completed_updates') != complete['maximum_updates']
            or manifest.get('maximum_updates') != complete['maximum_updates']):
        raise ValueError('A complete, non-smoke SFT run is required before distillation')
    if manifest.get('identity') != identity:
        raise ValueError('SFT data, Base model, source or settings differ; use a fresh CHECKPOINT_DIR')
    selected = (folder / 'best').resolve()
    if checkpoint.resolve() != selected or Path(complete['selected_checkpoint']).resolve() != selected:
        raise ValueError('Baseline must use the selected shared SFT checkpoint')
    if selected.parent != folder.resolve() or not (selected / 'config.json').is_file():
        raise ValueError('Selected SFT checkpoint is missing or outside its run directory')
    if not any(selected.glob('*.safetensors')) and not (selected / 'pytorch_model.bin').is_file():
        raise ValueError('Selected SFT model weights are missing')
    return selected


def ensure_sft(checkpoint, config, size, run):
    """All baselines share one complete SFT run per model size; failed SFT blocks KD."""
    from data.prepare import preparation_lock
    checkpoint = Path(checkpoint)
    expected = Path(config['checkpoint']['local_root']) / f'sft_{size}_seed42/best'
    identity = sft_identity(config)
    with preparation_lock(checkpoint.parent.with_suffix('.sft_stage')):
        if checkpoint.parent.exists():
            selected = verify_sft(checkpoint, identity)
            print('[sft] reuse ' + str(selected), flush=True)
            return selected
        if checkpoint.absolute() != expected.absolute():
            raise ValueError('External SFT checkpoint is missing; check SFT_MODEL_PATH or models.student')
        run()
        selected = verify_sft(checkpoint, identity)
        print('[sft] complete ' + str(selected), flush=True)
        return selected
