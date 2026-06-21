"""
Layer Normalization Mutant Generator
生成120个变异体，保存为独立py文件
"""

import os
import inspect

# 创建目录
OUTPUT_DIR = "mutants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 基础源程序模板
BASE_CODE = '''import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer Normalization - M{mid} ({op_type})"""
    def __init__(self, normalized_shape=64, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    
    def forward(self, x):
{forward_code}
        return x_norm
'''

# 原始程序forward代码 (M00)
ORIG_FORWARD = '''        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias'''

# ==================== ROR变异体 (关系运算符替换) M01-M12 ====================

ROR_MUTANTS = {
    "M01": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias  # Original"),
    "M02": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=True)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M03": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.eps > 1e-10:\n            x_norm = self.weight * x_norm + self.bias"),
    "M04": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.eps < 1e-3:\n            x_norm = self.weight * x_norm + self.bias"),
    "M05": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.eps >= 1e-8:\n            x_norm = self.weight * x_norm + self.bias"),
    "M06": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.eps <= 1e-4:\n            x_norm = self.weight * x_norm + self.bias"),
    "M07": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.eps == 1e-5:\n            x_norm = self.weight * x_norm + self.bias"),
    "M08": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.eps != 0:\n            x_norm = self.weight * x_norm + self.bias"),
    "M09": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        mask = (var > 0).float()\n        x_norm = (x - mean) / (torch.sqrt(var + self.eps) * mask + 1e-6)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M10": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        mask = (var < 1).float()\n        x_norm = (x - mean) / (torch.sqrt(var + self.eps) * mask + 1e-6)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M11": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if var.sum() > 0:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x - mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M12": ("ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if var.sum() >= 0:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x - mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
}

# ==================== AOR变异体 (算术运算符替换) M13-M22 ====================

AOR_MUTANTS = {
    "M13": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) + torch.sqrt(var + self.eps)  # / 改为 +\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M14": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) - torch.sqrt(var + self.eps)  # / 改为 -\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M15": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) * torch.sqrt(var + self.eps)  # / 改为 *\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M16": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x + mean) / torch.sqrt(var + self.eps)  # - 改为 +\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M17": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x * mean) / torch.sqrt(var + self.eps)  # - 改为 *\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M18": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x / mean) / torch.sqrt(var + self.eps)  # - 改为 / (危险)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M19": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var - self.eps)  # + 改为 -\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M20": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var * self.eps)  # + 改为 *\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M21": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / (var + self.eps)  # 去掉sqrt\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M22": ("AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.pow(var + self.eps, 0.25)  # sqrt改为pow 0.25\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
}

# ==================== LOR变异体 (逻辑运算符替换) M23-M30 ====================

LOR_MUTANTS = {
    "M23": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine and self.training:\n            x_norm = self.weight * x_norm + self.bias"),
    "M24": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine or self.training:\n            x_norm = self.weight * x_norm + self.bias"),
    "M25": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if not self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M26": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine and not self.training:\n            x_norm = self.weight * x_norm + self.bias"),
    "M27": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine or not self.training:\n            x_norm = self.weight * x_norm + self.bias"),
    "M28": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if not (self.elementwise_affine and self.training):\n            x_norm = self.weight * x_norm + self.bias"),
    "M29": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine ^ self.training:  # XOR\n            x_norm = self.weight * x_norm + self.bias"),
    "M30": ("LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if (self.elementwise_affine and self.training) or (not self.elementwise_affine and not self.training):\n            x_norm = self.weight * x_norm + self.bias"),
}

# ==================== COR变异体 (条件运算符替换) M31-M36 ====================

COR_MUTANTS = {
    "M31": ("COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.bias + self.weight * x_norm  # 交换顺序"),
    "M32": ("COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm - self.bias  # + 改为 -"),
    "M33": ("COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight / x_norm + self.bias  # * 改为 / (危险)"),
    "M34": ("COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * (x_norm + self.bias)  # 改变结合律"),
    "M35": ("COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = (self.weight * x_norm) + self.bias  # 显式括号"),
    "M36": ("COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            temp = self.weight * x_norm\n            x_norm = temp + self.bias\n        else:\n            x_norm = x_norm"),
}

# ==================== UOI变异体 (一元运算符插入) M37-M43 ====================

UOI_MUTANTS = {
    "M37": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = -(x - mean) / torch.sqrt(var + self.eps)  # 取负\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M38": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = -self.weight * x_norm + self.bias  # weight取负"),
    "M39": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + (-self.bias)  # bias取负"),
    "M40": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(-var + self.eps)  # var取负(危险)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M41": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(abs(var) + self.eps)  # abs(var)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M42": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        x_norm = -(-x_norm)  # 双重否定\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M43": ("UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * abs(x_norm) + self.bias  # abs(x_norm)"),
}

# ==================== SDL变异体 (语句删除) M44-M50 ====================

SDL_MUTANTS = {
    "M44": ("SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        # 删除affine变换\n        # if self.elementwise_affine:\n        #     x_norm = self.weight * x_norm + self.bias"),
    "M45": ("SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        # var = x.var(dim=-1, keepdim=True, unbiased=False)  # 删除var计算\n        var = torch.zeros_like(mean)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M46": ("SDL", "        # mean = x.mean(dim=-1, keepdim=True)  # 删除mean计算\n        mean = torch.zeros_like(x.mean(dim=-1, keepdim=True))\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M47": ("SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        # x_norm = (x - mean) / torch.sqrt(var + self.eps)  # 删除归一化\n        x_norm = x\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M48": ("SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm  # 删除bias\n            # + self.bias"),
    "M49": ("SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = x_norm + self.bias  # 删除weight\n            # self.weight * x_norm"),
    "M50": ("SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean)  # 删除除法 / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
}

# ==================== ABS变异体 (绝对值相关) M51-M60 ====================

ABS_MUTANTS = {
    "M51": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(torch.abs(var) + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M52": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + abs(self.eps))\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M53": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = torch.abs(x - mean) / torch.sqrt(var + self.eps)  # abs分子\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M54": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.abs(torch.sqrt(var + self.eps))  # abs分母\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M55": ("ABS", "        mean = torch.abs(x.mean(dim=-1, keepdim=True))  # abs(mean)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M56": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = torch.abs(x.var(dim=-1, keepdim=True, unbiased=False))  # abs(var)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M57": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = torch.abs(self.weight) * x_norm + self.bias  # abs(weight)\n            # x_norm = self.weight * x_norm + self.bias"),
    "M58": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + torch.abs(self.bias)  # abs(bias)\n            # x_norm = self.weight * x_norm + self.bias"),
    "M59": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = torch.abs((x - mean) / torch.sqrt(var + self.eps))  # abs整体\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M60": ("ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = torch.abs(self.weight * x_norm + self.bias)  # abs结果\n            # x_norm = self.weight * x_norm + self.bias"),
}

# ==================== 等价变异体 EQUI M61-M75 ====================

EQUI_MUTANTS = {
    "M61": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        x_norm = x_norm * 1.0  # 乘以1，等价\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M62": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.bias + self.weight * x_norm  # 交换加法顺序，等价"),
    "M63": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x + (-mean)) / torch.sqrt(var + self.eps)  # -mean 改为 +(-mean)，等价\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M64": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias + 0.0  # 加0，等价"),
    "M65": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        x_norm = -(-x_norm)  # 双重否定，等价\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M66": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * (x_norm * 1.0) + self.bias  # 嵌套乘以1，等价"),
    "M67": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            temp = x_norm\n            x_norm = self.weight * temp + self.bias  # 引入临时变量，等价"),
    "M68": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = (self.weight * x_norm) + (self.bias * 1.0)  # bias乘以1，等价"),
    "M69": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if True:  # 永真条件，等价\n            x_norm = self.weight * x_norm + self.bias"),
    "M70": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine and True:  # 添加永真合取，等价\n            x_norm = self.weight * x_norm + self.bias"),
    "M71": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias\n        # 添加死代码\n        else:\n            pass  # 永远不会执行，等价"),
    "M72": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        # 冗余自赋值\n        x_norm = x_norm\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    "M73": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            # 显式括号，不改变运算顺序\n            x_norm = ((self.weight * x_norm) + self.bias)"),
    "M74": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            w = self.weight\n            b = self.bias\n            x_norm = w * x_norm + b  # 变量别名，等价"),
    "M75": ("EQUI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = 1.0 * self.weight * x_norm + 0.0 + self.bias  # 1.0*和+0.0，等价"),
}

# ==================== 扩展变异体生成 (M76-M120) ====================

def generate_extended_mutants():
    """基于M01-M75的模式，每种算子再扩展10个"""
    extended = {}
    
    # ROR扩展 M76-M85 (基于M01-M12的模式变化)
    ror_extensions = [
        ("M76", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if var.min() > 0:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x - mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M77", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if var.max() < 100:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x - mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M78", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if x.shape[-1] == 64:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M79", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if mean.abs().mean() < 10:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M80", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if var.mean() != 0:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x - mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M81", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if self.eps in [1e-5, 1e-8]:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = (x - mean) / torch.sqrt(var + 1e-5)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M82", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if x.numel() > 100:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M83", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if x.dim() == 3:\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M84", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if torch.isfinite(var).all():\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = torch.zeros_like(x)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M85", "ROR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        if var.std() < mean.std():\n            x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        else:\n            x_norm = x - mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    ]
    
    # AOR扩展 M86-M95
    aor_extensions = [
        ("M86", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / (var + self.eps)  # 无sqrt\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M87", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.pow(var + self.eps, 0.5)  # pow 0.5\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M88", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) * torch.rsqrt(var + self.eps)  # rsqrt\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M89", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / (torch.sqrt(var) + self.eps)  # eps位置变化\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M90", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var) + self.eps  # eps移到外面\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M91", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps * 2)  # eps*2\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M92", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps / 2)  # eps/2\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M93", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x + mean) / torch.sqrt(var + self.eps)  # x+mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M94", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x * mean) / torch.sqrt(var + self.eps)  # x*mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M95", "AOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = x - mean + torch.sqrt(var + self.eps)  # 改为+sqrt\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    ]
    
    # LOR扩展 M96-M100
    lor_extensions = [
        ("M96", "LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine and x.is_cuda:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M97", "LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine or x.requires_grad:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M98", "LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if not (self.elementwise_affine or self.training):\n            x_norm = self.weight * x_norm + self.bias"),
        ("M99", "LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if (self.elementwise_affine and not self.training) or (not self.elementwise_affine and self.training):\n            x_norm = self.weight * x_norm + self.bias"),
        ("M100", "LOR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if bool(self.elementwise_affine) != bool(self.training):\n            x_norm = self.weight * x_norm + self.bias"),
    ]
    
    # COR扩展 M101-M105
    cor_extensions = [
        ("M101", "COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = (self.weight + 0.0) * x_norm + self.bias"),
        ("M102", "COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * (x_norm + 0.0) + self.bias"),
        ("M103", "COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = 0.0 + self.weight * x_norm + self.bias"),
        ("M104", "COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * 1.0 * x_norm + self.bias"),
        ("M105", "COR", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm * 1.0 + self.bias"),
    ]
    
    # UOI扩展 M106-M110
    uoi_extensions = [
        ("M106", "UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        x_norm = +x_norm  # 一元正号\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M107", "UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = (+self.weight) * x_norm + self.bias"),
        ("M108", "UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + (+self.bias)"),
        ("M109", "UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - (+mean)) / torch.sqrt(var + self.eps)  # +mean\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M110", "UOI", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt((+var) + self.eps)  # +var\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
    ]
    
    # SDL扩展 M111-M115
    sdl_extensions = [
        ("M111", "SDL", "        # mean = x.mean(dim=-1, keepdim=True)\n        mean = torch.zeros(1, 1, 64)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M112", "SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        # var = x.var(dim=-1, keepdim=True, unbiased=False)\n        var = torch.ones_like(mean) * 0.5\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M113", "SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        # x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        x_norm = torch.zeros_like(x)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M114", "SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        # if self.elementwise_affine:\n        #     x_norm = self.weight * x_norm + self.bias"),
        ("M115", "SDL", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            # x_norm = self.weight * x_norm + self.bias\n            x_norm = x_norm"),
    ]
    
    # ABS扩展 M116-M120
    abs_extensions = [
        ("M116", "ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(torch.clamp(var, min=0) + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M117", "ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + torch.abs(self.eps))\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M118", "ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = torch.clamp((x - mean) / torch.sqrt(var + self.eps), min=-10, max=10)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + self.bias"),
        ("M119", "ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = torch.clamp(self.weight, min=0) * x_norm + self.bias"),
        ("M120", "ABS", "        mean = x.mean(dim=-1, keepdim=True)\n        var = x.var(dim=-1, keepdim=True, unbiased=False)\n        x_norm = (x - mean) / torch.sqrt(var + self.eps)\n        if self.elementwise_affine:\n            x_norm = self.weight * x_norm + torch.clamp(self.bias, min=-1, max=1)"),
    ]
    
    all_extensions = (ror_extensions + aor_extensions + lor_extensions + 
                     cor_extensions + uoi_extensions + sdl_extensions + abs_extensions)
    
    for mid, op_type, code in all_extensions:
        extended[mid] = (op_type, code)
    
    return extended

# ==================== 主生成函数 ====================

def generate_all_mutants():
    """生成所有120个变异体"""
    
    # 合并所有基础变异体
    all_mutants = {}
    all_mutants.update({"M00": ("ORIG", ORIG_FORWARD)})
    all_mutants.update(ROR_MUTANTS)
    all_mutants.update(AOR_MUTANTS)
    all_mutants.update(LOR_MUTANTS)
    all_mutants.update(COR_MUTANTS)
    all_mutants.update(UOI_MUTANTS)
    all_mutants.update(SDL_MUTANTS)
    all_mutants.update(ABS_MUTANTS)
    all_mutants.update(EQUI_MUTANTS)
    
    # 添加扩展变异体
    extended = generate_extended_mutants()
    all_mutants.update(extended)
    
    # 验证数量
    print(f"Total mutants to generate: {len(all_mutants)}")
    print(f"Expected: M00-M120 = 121 files")
    
    # 生成文件
    generated = []
    for mid in sorted(all_mutants.keys()):
        op_type, forward_code = all_mutants[mid]
        filename = f"{mid}.py"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # 生成代码
        code = BASE_CODE.format(mid=mid, op_type=op_type, forward_code=forward_code)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        generated.append((mid, op_type, filename))
        print(f"Generated: {filename} ({op_type})")
    
    # 生成索引文件
    generate_index(generated)
    
    return generated

def generate_index(generated_list):
    """生成变异体索引文件"""
    index_content = "# Layer Normalization Mutants Index\n\n"
    index_content += "## Mutant List (M00-M120)\n\n"
    index_content += "| ID | Type | Description | File |\n"
    index_content += "|----|------|-------------|------|\n"
    
    type_descriptions = {
        "ORIG": "Original Program",
        "ROR": "Relational Operator Replacement",
        "AOR": "Arithmetic Operator Replacement",
        "LOR": "Logical Operator Replacement",
        "COR": "Conditional Operator Replacement",
        "UOI": "Unary Operator Insertion",
        "SDL": "Statement Deletion",
        "ABS": "Absolute Value Related",
        "EQUI": "Equivalent Mutant Candidate"
    }
    
    for mid, op_type, filename in generated_list:
        desc = type_descriptions.get(op_type, "Unknown")
        index_content += f"| {mid} | {op_type} | {desc} | {filename} |\n"
    
    index_content += f"\n## Statistics\n\n"
    index_content += f"- Total Mutants: {len(generated_list)}\n"
    
    # 统计各类型数量
    from collections import Counter
    type_counts = Counter([op for _, op, _ in generated_list])
    index_content += "- Type Distribution:\n"
    for op_type, count in sorted(type_counts.items()):
        index_content += f"  - {op_type}: {count}\n"
    
    index_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"\nGenerated index: {index_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Layer Normalization Mutant Generator")
    print("Generating M00-M120 (121 mutants)")
    print("=" * 60)
    
    generated = generate_all_mutants()
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Total files: {len(generated)}")
    print("=" * 60)