import torch


class FP32MasterAdamW(torch.optim.Optimizer):
    precision = 'fp32_master_cpu_v1'

    def __init__(self, parameters, *, lr, betas, weight_decay, eps=1e-8):
        parameters = [p for p in parameters if p.requires_grad]
        super().__init__(parameters, dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps))
        self._parameters = parameters
        self._masters = [p.detach().to(device='cpu', dtype=torch.float32, copy=True) for p in parameters]
        self._gradient_buffers = [None] * len(parameters)
        self._optimizer = torch.optim.AdamW(
            self._masters, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, foreach=True,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('FP32MasterAdamW expects gradients accumulated before step')
        for outer, inner in zip(self.param_groups, self._optimizer.param_groups):
            for key, value in outer.items():
                if key != 'params':
                    inner[key] = value
        for index, (parameter, master) in enumerate(zip(self._parameters, self._masters)):
            if parameter.grad is None:
                master.grad = None
                continue
            if parameter.grad.is_sparse:
                raise ValueError('sparse gradients are unsupported')
            if self._gradient_buffers[index] is None:
                self._gradient_buffers[index] = torch.empty_like(master)
            self._gradient_buffers[index].copy_(parameter.grad)
            master.grad = self._gradient_buffers[index]
        self._optimizer.step()
        for parameter, master in zip(self._parameters, self._masters):
            # Convert on CPU, avoiding a full FP32 weight temporary on the GPU.
            parameter.copy_(master.to(dtype=parameter.dtype))

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if hasattr(self, '_optimizer'):
            self._optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        for outer,inner in zip(self.param_groups,self._optimizer.param_groups):
            inner.update({key:value for key,value in outer.items() if key!='params'})
        result = self._optimizer.state_dict()
        result.update(optimizer_precision=self.precision, master_parameters=self._masters)
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get('optimizer_precision') != self.precision:
            raise ValueError('optimizer precision differs; start a separate run from Base')
        saved = state_dict['master_parameters']
        if len(saved) != len(self._masters):
            raise ValueError('master parameter count differs')
        with torch.no_grad():
            for master, value in zip(self._masters, saved):
                if value.shape != master.shape or value.dtype != torch.float32:
                    raise ValueError('master parameter identity differs')
                master.copy_(value)
        self._optimizer.load_state_dict({k: state_dict[k] for k in ['state', 'param_groups']})
        for outer, inner in zip(self.param_groups, self._optimizer.param_groups):
            outer.update({k: v for k, v in inner.items() if k != 'params'})
        with torch.no_grad():
            for parameter, master in zip(self._parameters, self._masters):
                parameter.copy_(master.to(dtype=parameter.dtype))
