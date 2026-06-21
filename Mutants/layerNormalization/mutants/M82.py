import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer Normalization - MM82 (ROR)"""
    def __init__(self, normalized_shape=64, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        if x.numel() > 100:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            x_norm = x
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm
