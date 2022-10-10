import torch
import torch.nn as nn
import transformers
from resources import _c

TORCH_OPTIM_CLASSES = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}

class BaseOptimizerBundle:
    def __init__(self, args):
        self.optim_class = TORCH_OPTIM_CLASSES.get(args.optim, None)
        self.lr = args.num_gpu*args.lr
        self.weight_decay = args.weight_decay
        self.optim_args = {
            'base': {'class': self.optim_class, 
            'params': [], 
            'is_func': lambda n, p: True, 
            'accum_steps': 1, 
            'args': {'lr': self.lr}}
        }
        self.step_count = 0
        self.optims = None
        self.schedulers = None

    def __str__(self):
        strings = []
        for k in self.optim_args.keys():
            strings.append(" ".join([_c(f'{self.optim_args[k]["class"]}({k}) ({self.optim_args[k]["args"]}, accum: {self.optim_args[k]["accum_steps"]}): ', attr='bold blue'), *[p[0] for p in self.optim_args[k]['params']]]))
        return '\n'.join(strings)

    def zero_grad(self):
        assert self.optims is not None
        for optim in self.optims.values(): optim.zero_grad(set_to_none=True)

    def _no_decay(self, n, p):
        return any([x in n.lower() for x in ['bias', 'layernorm', 'layer_norm']]) or p.shape == (1,)
    
    def inject_params(self, net: nn.Module):
        self.optims = {}
        self.step_count = 0
        for n,p in net.named_parameters():
            for k in self.optim_args.keys():
                if p.requires_grad and self.optim_args[k]['is_func'](n,p):
                    self.optim_args[k]['params'].append((n, p))

        for k, v in self.optim_args.items():
            grouped_params = [{'params': [p for n,p in v['params'] if self._no_decay(n,p)], 'weight_decay': 0}, 
                              {'params': [p for n,p in v['params'] if not self._no_decay(n,p)], 'weight_decay': 0.01}]
            if len(v['params']) > 0: 
                self.optims[k] = v['class'](grouped_params, **v['args'])

    def init_schedulers(self, args, per_epoch_steps: int):
        assert self.optims is not None
        total_steps = {k: per_epoch_steps*args.num_epochs/self.optim_args[k]['accum_steps'] for k in self.optims.keys()}
        self.schedulers = {k: transformers.get_linear_schedule_with_warmup(
            optim, 
            num_warmup_steps=int(args.warmup*total_steps[k]), 
            num_training_steps=total_steps[k]
        ) for k, optim in self.optims.items()}

    def step_and_zero_grad(self, scaler=None):
        assert self.optims is not None
        for k in self.optims.keys():
            if self.step_count%self.optim_args[k]["accum_steps"] == 0:
                if scaler is not None:
                    scaler.step(self.optims[k])
                    scaler.update()
                else:
                    self.optims[k].step()
                if self.schedulers: self.schedulers[k].step()
                self.optims[k].zero_grad(set_to_none=True)
        self.step_count += 1

class EliasOptimizerBundle(BaseOptimizerBundle):
    def __init__(self, args):
        super().__init__(args)
        self.numy = args.numy
        self.optim_args = {
            'tf': {'class': self.optim_class, 'params': [], 'is_func': self._is_tf_param, 'accum_steps': 1, 'args': {'lr': args.num_gpu*args.lr_tf}},
            'wl' : {'class': self.optim_class, 'params': [], 'is_func': self._is_wl_param, 'accum_steps': args.w_accumulation_steps, 'args': {'lr': args.num_gpu*args.lr_wl}},
            'other': {'class': self.optim_class, 'params': [], 'is_func': self._is_other_param, 'accum_steps': args.w_accumulation_steps, 'args': {'lr': args.num_gpu*args.lr}}
        }

    def _no_decay(self, n, p):
        return any([x in n.lower() for x in ['bias', 'layernorm', 'layer_norm']])

    def _is_tf_param(self, n, p):
        return n.startswith('tf.') or n.startswith('WL_transform.') or n.startswith('bottleneck.')

    def _is_wl_param(self, n, p):
        return any(abs(x - self.numy) <= 1 for x in p.shape) or n.startswith('A_nz_vals')

    def _is_other_param(self, n, p):
        return (not self._is_tf_param(n,p)) and (not self._is_wl_param(n,p))

OPTIM_BUNDLES = {
        'base': BaseOptimizerBundle,
        'elias': EliasOptimizerBundle
        }