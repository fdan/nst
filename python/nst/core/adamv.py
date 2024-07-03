from typing import List, Optional

# from nst.oiio import utils
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer,
                                   _use_grad_for_differentiable,
                                   _get_value,
                                   _stack_if_compiling,
                                   _dispatch_sqrt,
                                   _default_to_fused_or_foreach,
                                   _capturable_doc,
                                   _differentiable_doc,
                                   _foreach_doc,
                                   _fused_doc,
                                   _maximize_doc)

__all__ = ['AdamV', 'adamv']


class AdamV(Optimizer):
    def __init__(self,
                 params,
                 mask,
                 # masked_tensor,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.mask = mask
        self.iteration = 0
        # self.masked_tensor = masked_tensor

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        maximize=maximize,
                        foreach=foreach,
                        capturable=capturable,
                        differentiable=differentiable,
                        fused=fused)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if group['capturable'] or group['fused']
                        else torch.tensor(0.)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')
                state_steps.append(state['step'])

    @_use_grad_for_differentiable
    def step(self, closure=None):

        self._cuda_graph_capture_health_check()

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                self.mask,
                self.iteration,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                eps=group['eps'],
                maximize=group['maximize']
            )

        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         mask: Tensor,
         iteration: int,
         beta1: float,
         beta2: float,
         lr: float,
         eps: float,
         maximize: bool):

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        step = _get_value(step_t)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        # Maintains the maximum of all 2nd moment running avg. till now
        torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

        # Use the max. for normalizing running avg. of gradient
        denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)

        # real_grad = torch.div(exp_avg, denom).mul(-step_size)
        # masked_grad = real_grad * mask
        # param += masked_grad

        # grad = torch.div(exp_avg, denom).mul(-step_size)
        # grad_buf = utils.tensor_to_buf(grad, raw=True)
        # grad_fp = '/mnt/ala/research/danielf/proj/seq/id01/id01_060/render/nst/style08/fgBase/v003/2068x876/exr/tc/animTest01/301/out/adam/grad.%04d.exr' % iteration
        # utils.write_exr(grad_buf, grad_fp)
        # # new_param = param + grad
        # # param_diff = new_param - param
        # # masked_param_diff = param_diff * mask
        # # param += masked_param_diff
        #
        # param_start_buf = utils.tensor_to_buf(param, colorspace='acescg')
        # param_start_fp = '/mnt/ala/research/danielf/proj/seq/id01/id01_060/render/nst/style08/fgBase/v003/2068x876/exr/tc/animTest01/301/out/adam/param_starting.%04d.exr' % iteration
        # utils.write_exr(param_start_buf, param_start_fp)
        #
        # param.addcdiv_(exp_avg, denom, value=-step_size)
        #
        # # write out param here
        # # from nst.core import utils
        # # utils.write_tensor(param, '/mnt/ala/research/danielf/proj/seq/id01/id01_060/render/nst/style08/fgBase/v003/2068x876/exr/tc/animTest01/301/out/out20_adamv.%03d.pt' % self.iteration)
        #
        #
        # param_final_buf = utils.tensor_to_buf(param, colorspace='acescg')
        # param_final_fp = '/mnt/ala/research/danielf/proj/seq/id01/id01_060/render/nst/style08/fgBase/v003/2068x876/exr/tc/animTest01/301/out/adam/param_final.%04d.exr' % iteration
        # utils.write_exr(param_final_buf, param_final_fp)
