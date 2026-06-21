"""
RMSNorm Mutant - M97
Generated: 2026-02-25 21:56:06
Mutation Operator: RMS
Description: weight减法
Original: self.weight * x_norm
Mutated: x_norm - self.weight
"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    参考实现来源：
    [1] 原始论文官方实现: https://github.com/bzhangGo/rmsnorm 
    [2] LLaMA-1/2/3 官方实现: https://github.com/facebookresearch/llama/blob/main/llama/model.py#L75 
    [3] HuggingFace Transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75 
    
    论文引用:
    Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. 
    NeurIPS 2019. https://arxiv.org/abs/1910.07467 
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        """
        Args:
            hidden_size: 隐藏层维度 (如 4096, 5120, 8192 等)
            eps: 防止除零的小数值 (LLaMA使用1e-6，比LayerNorm的1e-5更保守)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # LLaMA中通常设为1e-6
    
    def forward(self, x):
        # 与LayerNorm的本质差异1: 不去均值，仅计算RMS
        # variance = mean(x^2), 而非 LayerNorm的variance = mean((x-mean)^2)
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # 与LayerNorm的本质差异2: 使用x而非(x-mean)进行归一化
        x_norm = x * torch.rsqrt(variance + self.variance_epsilon)
        
        # 与LayerNorm的本质差异3: 只有weight，没有bias
        return x_norm - self.weight

    def extra_repr(self):
        return f'hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}'


# 使用示例 (与LLaMA-2 7B配置一致)
if __name__ == "__main__":
    # LLaMA-2 7B: hidden_size=4096, eps=1e-6
    rms_norm = RMSNorm(hidden_size=4096, eps=1e-6)
    
    # 模拟输入: (batch_size=2, seq_len=8, hidden_dim=4096)
    x = torch.randn(2, 8, 4096)
    
    # 前向传播
    output = rms_norm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {rms_norm.weight.shape}")  # 应该是 [4096]
    
    # 数值稳定性测试: 全零输入 (RMSNorm的危险边界)
    x_zero = torch.zeros(2, 8, 4096)
    output_zero = rms_norm(x_zero)
    print(f"\nZero input test (should be 0, not NaN): {output_zero.abs().max().item()}")