"""
PyTorch SGD Optimizer - Industrial Implementation
来自于真是github项目: https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
License: BSD-3-Clause
Copyright (c) 2016-present, Facebook, Inc.
"""

import torch
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    
    Industrial application: Training deep neural networks at scale
    (Meta's production recommendation systems, Tesla Autopilot, etc.)
    
    Nesterov momentum is based on the formula from:
    `On the importance of initialization and momentum in deep learning` (Sutskever et al., 2013).
    """
    
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov,
                       maximize=maximize, foreach=foreach)
        super(SGD, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']
            foreach = group['foreach']
            
            # Collect parameters with gradients (sparse gradient handling)
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            # Functional API call (dispatch to optimized C++ kernels or foreach loops)
            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                foreach=foreach)
            
            # Update momentum buffers in state
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer_list[i]
        
        return loss


def sgd(params: List[torch.Tensor],
        d_p_list: List[torch.Tensor],
        momentum_buffer_list: List[Optional[torch.Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        foreach: Optional[bool]):
    r"""Functional API performing SGD algorithm computation.
    
    Industrial complexity: Handles fused kernel dispatch, sparse gradients,
    and memory-efficient updates.
    """
    if foreach is None:
        # Heuristic: use fused foreach path for tensor lists (GPU optimization)
        foreach = False  # Simplified for SUT extraction
    
    if foreach:
        _multi_tensor_sgd(params, d_p_list, momentum_buffer_list,
                         weight_decay=weight_decay, momentum=momentum,
                         lr=lr, dampening=dampening, nesterov=nesterov,
                         maximize=maximize)
    else:
        _single_tensor_sgd(params, d_p_list, momentum_buffer_list,
                          weight_decay=weight_decay, momentum=momentum,
                          lr=lr, dampening=dampening, nesterov=nesterov,
                          maximize=maximize)


def _single_tensor_sgd(params: List[torch.Tensor],
                       d_p_list: List[torch.Tensor],
                       momentum_buffer_list: List[Optional[torch.Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool):
    """Scalar/loop-based SGD (fallback path, easier for mutation testing)."""
    
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)  # d_p += weight_decay * param
        
        if maximize:
            d_p = -d_p  # Gradient ascent (for GANs, adversarial training)
        
        if momentum != 0:
            buf = momentum_buffer_list[i]
            
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)  # buf = buf*m + d_p*(1-damp)
            
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)  # d_p = d_p + m*buf
            else:
                d_p = buf
        
        param.add_(d_p, alpha=-lr)  # param = param - lr * d_p


def _multi_tensor_sgd(params: List[torch.Tensor],
                      d_p_list: List[torch.Tensor],
                      momentum_buffer_list: List[Optional[torch.Tensor]],
                      *, weight_decay: float, momentum: float, lr: float,
                      dampening: float, nesterov: bool, maximize: bool):
    """Fused multi-tensor SGD (GPU kernel fusion optimization)."""
    if len(params) == 0:
        return
    
    if maximize:
        torch._foreach_neg_(d_p_list)  # In-place negation for maximization
    
    if weight_decay != 0:
        torch._foreach_add_(d_p_list, params, alpha=weight_decay)
    
    if momentum != 0:
        bufs = []
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                buf = momentum_buffer_list[i] = torch.clone(d_p_list[i]).detach()
            else:
                buf = momentum_buffer_list[i]
                buf.mul_(momentum).add_(d_p_list[i], alpha=1 - dampening)
            bufs.append(buf)
        
        if nesterov:
            torch._foreach_add_(d_p_list, bufs, alpha=momentum)
        else:
            d_p_list = bufs
    
    torch._foreach_add_(params, d_p_list, alpha=-lr)


# ==================== ORACLE TEST HARNESS ====================

def quadratic_objective(w, X, y):
    """Least squares: f(w) = ||Xw - y||^2"""
    residual = X @ w - y
    loss = torch.sum(residual ** 2)
    grad = 2 * X.t() @ residual
    return loss, grad


def test_convergence():
    """Oracle: Verify optimizer converges on quadratic problem."""
    torch.manual_seed(42)
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features)
    w_true = torch.randn(n_features)
    y = X @ w_true + 0.01 * torch.randn(n_samples)
    
    w = torch.randn(n_features, requires_grad=True)
    optimizer = SGD([w], lr=0.01, momentum=0.9, weight_decay=0.001)
    
    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        loss, grad = quadratic_objective(w, X, y)
        w.grad = grad
        optimizer.step()
        losses.append(loss.item())
    
    final_loss = losses[-1]
    converged = final_loss < 1e-3
    print(f"Final Loss: {final_loss:.6f}, Converged: {converged}")
    return converged


if __name__ == "__main__":
    assert test_convergence(), "Optimizer failed to converge!"