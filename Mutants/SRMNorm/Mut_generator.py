import os
# 保存完整的生成器代码到文件
generator_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSNorm 变异体自动生成器
基于M00.py生成110个变异体 (M01.py - M110.py)
支持17类变异算子
"""

import os
from datetime import datetime

# 原始程序代码 (M00.py)
SOURCE_CODE = \'\'\'import torch
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
        return self.weight * x_norm

    def extra_repr(self):
        return f\'hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}\'


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
    print(f"\\\\nZero input test (should be 0, not NaN): {output_zero.abs().max().item()}")\'\'\'

# 定义110个变异算子
MUTATION_OPERATORS = {
    # === 算术运算符变异 (AOR) ===
    "M01": ("AOR", "+->-", "variance + self.variance_epsilon", "variance - self.variance_epsilon"),
    "M02": ("AOR", "+->*", "variance + self.variance_epsilon", "variance * self.variance_epsilon"),
    "M03": ("AOR", "+->/", "variance + self.variance_epsilon", "variance / self.variance_epsilon"),
    "M04": ("AOR", "*->+", "x * torch.rsqrt", "x + torch.rsqrt"),
    "M05": ("AOR", "*->-", "x * torch.rsqrt", "x - torch.rsqrt"),
    "M06": ("AOR", "*->/", "x * torch.rsqrt", "x / torch.rsqrt"),
    "M07": ("AOR", "*->+", "self.weight * x_norm", "self.weight + x_norm"),
    "M08": ("AOR", "*->-", "self.weight * x_norm", "self.weight - x_norm"),
    "M09": ("AOR", "*->/", "self.weight * x_norm", "self.weight / x_norm"),
    "M10": ("AOR", "pow(2)->pow(3)", "x.pow(2)", "x.pow(3)"),
    "M11": ("AOR", "pow(2)->abs()", "x.pow(2)", "x.abs()"),
    "M12": ("AOR", "pow(2)->**2", "x.pow(2)", "x**2"),
    
    # === 关系运算符变异 (ROR) ===
    "M13": ("ROR", "keepdim=True->False", "keepdim=True", "keepdim=False"),
    "M14": ("ROR", "mean(-1->-2", "mean(-1, keepdim=True)", "mean(-2, keepdim=True)"),
    "M15": ("ROR", "mean(-1->0", "mean(-1, keepdim=True)", "mean(0, keepdim=True)"),
    "M16": ("ROR", "mean->sum", ".mean(-1", ".sum(-1"),
    "M17": ("ROR", "rsqrt->sqrt", "torch.rsqrt", "torch.sqrt"),
    "M18": ("ROR", "rsqrt->reciprocal", "torch.rsqrt", "torch.reciprocal"),
    "M19": ("ROR", "ones->zeros", "torch.ones", "torch.zeros"),
    "M20": ("ROR", "ones->randn", "torch.ones", "torch.randn"),
    
    # === 常量变异 (CR) ===
    "M21": ("CR", "1e-6->1e-5", "eps=1e-6", "eps=1e-5"),
    "M22": ("CR", "1e-6->1e-7", "eps=1e-6", "eps=1e-7"),
    "M23": ("CR", "1e-6->1e-4", "eps=1e-6", "eps=1e-4"),
    "M24": ("CR", "1e-6->0", "eps=1e-6", "eps=0"),
    "M25": ("CR", "1e-6->1", "eps=1e-6", "eps=1"),
    "M26": ("CR", "pow(2)->pow(3)", "x.pow(2)", "x.pow(3)"),
    "M27": ("CR", "4096->2048", "hidden_size=4096", "hidden_size=2048"),
    "M28": ("CR", "4096->8192", "hidden_size=4096", "hidden_size=8192"),
    "M29": ("CR", "4096->1024", "hidden_size=4096", "hidden_size=1024"),
    "M30": ("CR", "batch=2->4", "torch.randn(2, 8", "torch.randn(4, 8"),
    
    # === 语句删除 (SDL) ===
    "M31": ("SDL", "删除keepdim", "keepdim=True", ""),
    "M32": ("SDL", "删除eps", " + self.variance_epsilon", ""),
    "M33": ("SDL", "删除weight乘法", "self.weight * ", ""),
    "M34": ("SDL", "删除rsqrt", "torch.rsqrt", ""),
    "M35": ("SDL", "删除pow(2)", ".pow(2)", ""),
    "M36": ("SDL", "删除mean", ".mean(-1, keepdim=True)", ""),
    
    # === 变量替换 (VVR) ===
    "M37": ("VVR", "x->x_norm", "x * torch.rsqrt", "x_norm * torch.rsqrt"),
    "M38": ("VVR", "variance->x", "variance + self", "x + self"),
    "M39": ("VVR", "x_norm->variance", "return self.weight * x_norm", "return self.weight * variance"),
    "M40": ("VVR", "weight->variance", "self.weight * x_norm", "self.variance * x_norm"),
    
    # === 函数调用变异 (FCR) ===
    "M41": ("FCR", "rsqrt->sqrt", "torch.rsqrt", "torch.sqrt"),
    "M42": ("FCR", "mean->sum", ".mean(-1", ".sum(-1"),
    "M43": ("FCR", "pow->abs", ".pow(2)", ".abs()"),
    "M44": ("FCR", "pow->square", ".pow(2)", "**2"),
    "M45": ("FCR", "ones->full", "torch.ones(hidden_size)", "torch.full((hidden_size,), 1.0)"),
    "M46": ("FCR", "randn->rand", "torch.randn", "torch.rand"),
    "M47": ("FCR", "rsqrt->pow(-0.5)", "torch.rsqrt(variance + self.variance_epsilon)", "torch.pow(variance + self.variance_epsilon, -0.5)"),
    "M48": ("FCR", "mean->var", ".mean(-1", ".var(-1"),
    "M49": ("FCR", "rsqrt->rsqrt+1", "torch.rsqrt(variance + self.variance_epsilon)", "torch.rsqrt(variance + self.variance_epsilon) + 1"),
    "M50": ("FCR", "pow(2)->pow(1)", "x.pow(2)", "x.pow(1)"),
    
    # === 赋值变异 (ASR) ===
    "M51": ("ASR", "=->+=", "variance =", "variance +="),
    "M52": ("ASR", "=->-=", "x_norm =", "x_norm -="),
    "M53": ("ASR", "=->*=", "x_norm =", "x_norm *="),
    "M54": ("ASR", "=->/=", "variance =", "variance /="),
    
    # === 边界值变异 (BVR) ===
    "M55": ("BVR", "eps=0", "eps=1e-6", "eps=0"),
    "M56": ("BVR", "eps=1", "eps=1e-6", "eps=1"),
    "M57": ("BVR", "eps=1e-10", "eps=1e-6", "eps=1e-10"),
    "M58": ("BVR", "hidden_size=1", "hidden_size=4096", "hidden_size=1"),
    "M59": ("BVR", "hidden_size=100000", "hidden_size=4096", "hidden_size=100000"),
    "M60": ("BVR", "batch=1", "torch.randn(2,", "torch.randn(1,"),
    
    # === 交换变异 (SOR) ===
    "M61": ("SOR", "交换乘数", "x * torch.rsqrt", "torch.rsqrt(variance + self.variance_epsilon) * x"),
    "M62": ("SOR", "交换weight", "self.weight * x_norm", "x_norm * self.weight"),
    "M63": ("SOR", "交换加法", "variance + self.variance_epsilon", "self.variance_epsilon + variance"),
    
    # === 一元运算符变异 (UOI) ===
    "M64": ("UOI", "+x->-x", "return self.weight * x_norm", "return -self.weight * x_norm"),
    "M65": ("UOI", "x->abs(x)", "x * torch.rsqrt", "x.abs() * torch.rsqrt"),
    "M66": ("UOI", "x->-abs(x)", "return self.weight * x_norm", "return -(self.weight * x_norm)"),
    "M67": ("UOI", "添加负号", "x_norm = x", "x_norm = -x"),
    
    # === 数据类型变异 (DTR) ===
    "M68": ("DTR", "float32->float64", "torch.randn(2, 8, 4096)", "torch.randn(2, 8, 4096, dtype=torch.float64)"),
    "M69": ("DTR", "float32->float16", "torch.randn(2, 8, 4096)", "torch.randn(2, 8, 4096, dtype=torch.float16)"),
    "M70": ("DTR", "Parameter->Tensor", "nn.Parameter(torch.ones", "torch.ones"),
    
    # === 异常处理变异 (EHR) ===
    "M71": ("EHR", "添加try-except", "variance = x.pow(2)", "try:\\\\n            variance = x.pow(2)\\\\n        except:\\\\n            variance = x"),
    "M72": ("EHR", "添加if-else", "variance = x.pow(2)", "if True:\\\\n            variance = x.pow(2)\\\\n        else:\\\\n            variance = x"),
    
    # === 返回值变异 (RVR) ===
    "M73": ("RVR", "return->return None", "return self.weight * x_norm", "return None"),
    "M74": ("RVR", "return->return x", "return self.weight * x_norm", "return x"),
    "M75": ("RVR", "return->return variance", "return self.weight * x_norm", "return variance"),
    
    # === 循环变异 (LVR) ===
    "M76": ("LVR", "添加循环", "output = rms_norm(x)", "for _ in range(1):\\\\n            output = rms_norm(x)"),
    
    # === 条件边界变异 (CBR) ===
    "M78": ("CBR", "if删除", "if __name__", "# if __name__"),
    "M79": ("CBR", "条件反转", "if __name__ == \\"__main__\\":", "if __name__ != \\"__main__\\":"),
    
    # === 字符串变异 (STR) ===
    "M80": ("STR", "修改repr", "RMSNorm", "RMS_Norm"),
    "M81": ("STR", "修改shape", "shape", "size"),
    
    # === 索引变异 (IOR) ===
    "M82": ("IOR", "-1->-2", "mean(-1", "mean(-2"),
    "M83": ("IOR", "-1->0", "mean(-1", "mean(0"),
    "M84": ("IOR", "-1->1", "mean(-1", "mean(1"),
    
    # === 位运算变异 (BOR) ===
    "M85": ("BOR", "&->|", "&", "|"),
    "M87": ("BOR", "^->&", "^", "&"),
    
    # === 特殊RMSNorm相关变异 == =
    "M90": ("RMS", "添加mean减法", "        variance = x.pow(2).mean(-1, keepdim=True)", "        mean = x.mean(-1, keepdim=True); variance = (x - mean).pow(2).mean(-1, keepdim=True)"),
    "M91": ("RMS", "替换为LayerNorm", "        x_norm = x * torch.rsqrt(variance + self.variance_epsilon)", "        mean = x.mean(-1, keepdim=True); x_norm = (x - mean) * torch.rsqrt(variance + self.variance_epsilon)"),
    "M92": ("RMS", "删除sqrt", "torch.rsqrt", "1/(variance + self.variance_epsilon)"),
    "M93": ("RMS", "双重rsqrt", "torch.rsqrt(variance + self.variance_epsilon)", "torch.rsqrt(torch.rsqrt(variance + self.variance_epsilon))"),
    "M94": ("RMS", "sqrt代替rsqrt", "torch.rsqrt", "1/torch.sqrt(variance + self.variance_epsilon)"),
    "M95": ("RMS", "添加bias", "return self.weight * x_norm", "return self.weight * x_norm + torch.ones_like(x_norm)"),
    "M96": ("RMS", "weight除法", "self.weight * x_norm", "x_norm / self.weight"),
    "M97": ("RMS", "weight减法", "self.weight * x_norm", "x_norm - self.weight"),
    "M98": ("RMS", "交换weight和x", "self.weight * x_norm", "x_norm * self.weight"),
    "M99": ("RMS", "mean维度错误", "mean(-1", "mean(1"),
    
    # === 数值精度变异 (NPR) ===
    "M100": ("NPR", "epsilon放大", "self.variance_epsilon", "self.variance_epsilon * 1e6"),
    "M101": ("NPR", "epsilon缩小", "self.variance_epsilon)", "self.variance_epsilon / 1e6)"),
    "M102": ("NPR", "添加噪声", "x * torch.rsqrt", "(x + 1e-8) * torch.rsqrt"),
    
    # === 内存访问变异 (MAR) ===
    "M103": ("MAR", "原地操作", "x_norm = x", "x_norm = x.clone()"),
    "M104": ("MAR", "detach", "x * torch.rsqrt", "x.detach() * torch.rsqrt"),
    
    # === 设备相关变异 (DVR) ===
    "M106": ("DVR", "cpu->cuda", "torch.randn(2, 8, 4096)", "torch.randn(2, 8, 4096).cuda()"),
    "M107": ("DVR", "添加device", "torch.randn(2, 8, 4096)", "torch.randn(2, 8, 4096, device=\'cpu\')"),
    
    # === 广播变异 (BCR) ===
    "M108": ("BCR", "删除keepdim", "keepdim=True", "keepdim=False"),
    "M109": ("BCR", "维度不匹配", "(variance + self.variance_epsilon)", "(variance.squeeze() + self.variance_epsilon)"),
    
    # === 梯度相关变异 (GDR) ===
    "M110": ("GDR", "no_grad", "def forward(self, x):", "@torch.no_grad()\\\\n    def forward(self, x):"),
}

def apply_mutation(code, mutation_id, mutation_info):
    """应用单个变异算子到代码"""
    operator_type, desc, old, new = mutation_info
    
    # 处理换行符转义
    new = new.replace("\\\\n", "\\n")
    old = old.replace("\\\\n", "\\n")
    
    # 只替换第一次出现
    if old in code:
        return code.replace(old, new, 1)
    return code

def generate_mutant_file(mutant_id, code, mutation_info):
    """生成单个变异体文件内容"""
    operator_type, desc, old, new = mutation_info
    
    header = f\'\'\'"""
RMSNorm Mutant - {mutant_id}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Mutation Operator: {operator_type}
Description: {desc}
Original: {old}
Mutated: {new}
"""

\'\'\'
    return header + code

def main():
    """主函数：生成110个变异体"""
    
    output_dir = os.getcwd()
    
    print("=" * 60)
    print("RMSNorm 变异体自动生成器")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"变异体数量: 110")
    print("=" * 60)
    
    generated_count = 0
    failed_mutations = []
    
    for i in range(1, 111):
        mutant_id = f"M{i:02d}"
        
        if mutant_id not in MUTATION_OPERATORS:
            print(f"警告: {mutant_id} 未定义，跳过")
            continue
        
        mutation_info = MUTATION_OPERATORS[mutant_id]
        
        try:
            # 应用变异
            mutated_code = apply_mutation(SOURCE_CODE, mutant_id, mutation_info)
            
            # 检查是否真的发生了变异
            if mutated_code == SOURCE_CODE:
                failed_mutations.append((mutant_id, "未发生变异"))
                continue
            
            # 生成文件内容
            file_content = generate_mutant_file(mutant_id, mutated_code, mutation_info)
            
            # 写入文件
            filename = os.path.join(output_dir, f"{mutant_id}.py")
            with open(filename, \'w\', encoding=\'utf-8\') as f:
                f.write(file_content)
            
            generated_count += 1
            print(f"✓ 已生成: {mutant_id}.py ({mutation_info[0]} - {mutation_info[1]})")
            
        except Exception as e:
            failed_mutations.append((mutant_id, str(e)))
            print(f"✗ 失败: {mutant_id} - {e}")
    
    print("=" * 60)
    print(f"生成完成: {generated_count}/110 个变异体")
    
    if failed_mutations:
        print(f"\\n失败列表 ({len(failed_mutations)}个):")
        for mid, reason in failed_mutations:
            print(f"  - {mid}: {reason}")
    
    # 生成变异体清单
    readme_content = generate_readme()
    readme_path = os.path.join(output_dir, "MUTANTS_README.md")
    with open(readme_path, \'w\', encoding=\'utf-8\') as f:
        f.write(readme_content)
    print(f"\\n✓ 已生成变异体清单: MUTANTS_README.md")
    
    print("=" * 60)

def generate_readme():
    """生成变异体说明文档"""
    return """# RMSNorm 变异体清单

## 生成信息
- 源程序: M00.py (RMSNorm)
- 变异体数量: 110
- 生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

## 变异算子分类

### 1. 算术运算符变异 (AOR) - M01-M12
- M01-M03: 加法运算符变异 (+ → -, *, /)
- M04-M06: 乘法运算符变异 (* → +, -, /)
- M07-M09: weight乘法变异
- M10-M12: pow(2)变异

### 2. 关系运算符变异 (ROR) - M13-M20
- M13: keepdim参数变异
- M14-M15: mean维度变异
- M16: mean→sum
- M17-M18: rsqrt变异
- M19-M20: 初始化变异

### 3. 常量变异 (CR) - M21-M30
- M21-M25: epsilon值变异
- M26-M30: 维度大小变异

### 4. 语句删除 (SDL) - M31-M36
- M31: 删除keepdim
- M32: 删除epsilon
- M33: 删除weight乘法
- M34: 删除rsqrt
- M35: 删除pow(2)
- M36: 删除mean

### 5. 变量替换 (VVR) - M37-M40
- M37-M40: 变量名替换

### 6. 函数调用变异 (FCR) - M41-M50
- M41-M50: 函数替换

### 7. 交换变异 (SOR) - M61-M63
- M61-M63: 操作数交换

### 8. 一元运算符变异 (UOI) - M64-M67
- M64-M67: 添加/修改一元运算符

### 9. 数据类型变异 (DTR) - M68-M70
- M68-M70: 数据类型修改

### 10. 返回值变异 (RVR) - M73-M75
- M73-M75: return语句变异

### 11. 索引变异 (IOR) - M82-M84
- M82-M84: 维度索引变异

### 12. RMSNorm特定变异 (RMS) - M90-M99
- M90-M91: LayerNorm化变异
- M92-M94: rsqrt操作变异
- M95: 添加bias
- M96-M99: weight操作变异

### 13. 数值精度变异 (NPR) - M100-M102
- M100-M102: epsilon缩放和噪声

### 14. 内存访问变异 (MAR) - M103-M104
- M103-M104: 内存操作变异

### 15. 设备相关变异 (DVR) - M106-M107
- M106-M107: 设备指定变异

### 16. 广播变异 (BCR) - M108-M109
- M108-M109: 广播行为变异

### 17. 梯度变异 (GDR) - M110
- M110: no_grad装饰器

## 使用方法
```bash
# 运行原始程序
python M00.py

# 运行变异体
python M01.py
python M02.py
...
python M110.py
```

## 等价变异体提示
以下变异体可能是等价的（需验证）：
- M12 (pow(2) → **2): 功能等价
- M62 (交换律): 乘法交换律
- M63 (加法交换律): epsilon交换

## 参考文献
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019.
- LLaMA官方实现: https://github.com/facebookresearch/llama
"""

if __name__ == "__main__":
    main()
'''

# 保存生成器代码
output_path = "rmsnorm_mutant_generator.py"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(generator_code)

print(f"✓ 生成器代码已保存到: {output_path}")
print(f"✓ 文件大小: {os.path.getsize(output_path)} bytes")
print("\n使用方法:")
print("1. 将M00.py放在同一目录下")
print("2. 运行: python rmsnorm_mutant_generator.py")
print("3. 将自动生成 M01.py - M110.py")
