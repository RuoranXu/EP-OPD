import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def backend_config(config, prepare_data=False):
    from evidence_opd.config import load_config
    backend = OmegaConf.merge(OmegaConf.create(load_config(ROOT / config.method.backend_config)),
                              config.backend_overrides)
    size = config.trainer.size
    if size not in {'4b', '1p7b'}:
        raise ValueError('trainer.size must be 4b or 1p7b')
    base = 'models/Qwen3-' + ('4B' if size == '4b' else '1.7B') + '-Base'
    if 'STUDENT_MODEL_PATH' not in os.environ and OmegaConf.select(config.backend_overrides, 'models.student_tokenizer') is None:
        backend.models.student_tokenizer = base
        if 'base_student' in backend.models:
            backend.models.base_student = base
        if config.method.entry == 'baseline' and OmegaConf.select(config.backend_overrides, 'models.student') is None:
            backend.models.student = base
    if config.method.entry != 'baseline' and 'SFT_MODEL_PATH' not in os.environ and OmegaConf.select(config.backend_overrides, 'models.student') is None:
        backend.models.student = f'outputs/checkpoints/sft_{size}_seed42/best'
    if config.method.entry == 'baseline' and 'EMA_MODEL_PATH' not in os.environ:
        if OmegaConf.select(config.backend_overrides, 'models.ema_initial_teacher') is None:
            backend.models.ema_initial_teacher = 'models/Qwen3-' + ('4B' if size == '4b' else '1.7B')
            backend.models.ema_tokenizer = backend.models.ema_initial_teacher
    name = f'{config.method.name}_{size}_seed{config.trainer.seed}'
    backend.experiment.seed = int(config.trainer.seed)
    backend.run.name = name
    backend = OmegaConf.to_container(backend, resolve=True)
    if prepare_data:
        from data.prepare import prepare_training
        prepare_training(backend, config.method.entry)
    path = ROOT / 'outputs/configs' / (name + '.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(backend)), encoding='utf8')
    return path, name


@hydra.main(version_base=None, config_path='config', config_name='main')
def main(config: DictConfig):
    import ray
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from workers.exact_worker import OPDWorker

    if config.trainer.nnodes != 1:
        raise ValueError('this exact-logit recipe currently supports one shared-filesystem node')
    if config.method.entry == 'sft' and config.trainer.seed != 42:
        raise ValueError('the exported SFT schedule is frozen at seed 42')
    path, _ = backend_config(config, prepare_data=True)
    args = ['--config', str(path)]
    if config.method.entry != 'baseline':
        args += ['--size', config.trainer.size]
    if config.method.entry == 'ep_opd':
        args += ['--seed', str(config.trainer.seed), '--run-prefix', config.method.name]
    if config.trainer.smoke_updates:
        if config.method.entry == 'baseline':
            raise ValueError('use backend_overrides.training.max_optimizer_steps for a baseline smoke')
        args += ['--smoke-updates', str(config.trainer.smoke_updates)]
    started = not ray.is_initialized()
    group = None
    try:
        if started:
            ray.init(address=config.ray.address, runtime_env={'env_vars': {
                'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'VLLM_USE_V1': '0', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'}})
        pool = RayResourcePool([int(config.method.workers)], use_gpu=True,
                               max_colocate_count=1, name_prefix='ep_opd')
        group = RayWorkerGroup(resource_pool=pool,
            ray_cls_with_init=RayClassWithInitArgs(ray.remote(OPDWorker)),
            name_prefix='ep_opd_' + config.method.name, bin_pack=True)
        mapping = group.describe()
        if len({item['node_id'] for item in mapping}) != 1:
            raise ValueError('all TP/rollout workers must occupy the same node')
        devices = ','.join(item['visible_devices'] for item in mapping)
        ray.get(group.execute(str(ROOT), config.method.entry, args, devices))
    finally:
        if group is not None:
            for worker in group._workers:
                ray.kill(worker, no_restart=True)
        if started:
            ray.shutdown()


if __name__ == '__main__':
    main()
