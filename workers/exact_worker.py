import os
from pathlib import Path
import runpy
import sys

import ray
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register


class OPDWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def describe(self):
        return {'rank': self.rank, 'world_size': self.world_size,
                'node_id': ray.get_runtime_context().get_node_id(),
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def execute(self, root, entry, arguments, group_devices):
        import torch.distributed as dist
        entries = {'ep_opd': 'trainer/ep_opd.py', 'baseline': 'trainer/baseline.py',
                   'sft': 'trainer/sft.py'}
        if entry not in entries:
            raise ValueError('unknown training entry')
        # VERL/Ray exposes one assigned CUDA device in each actor. Global RANK
        # remains 0..N-1 for the student/TP-teacher/rollout process groups.
        os.environ.update(LOCAL_RANK='0', EP_OPD_VERL_WORKER='1',
                          EP_OPD_GROUP_DEVICES=group_devices)
        os.environ.pop('DEEPSEEK_API_KEY', None)
        root = Path(root).resolve()
        os.chdir(root)
        sys.path[:0] = [str(root / 'src'), str(root)]
        saved = sys.argv
        sys.argv = [str(root / entries[entry]), *arguments]
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError('training entry exited with code ' + str(exc.code)) from exc
        finally:
            sys.argv = saved
            if dist.is_initialized():
                dist.destroy_process_group()
        return {'rank': self.rank, 'entry': entry, 'complete': True}
