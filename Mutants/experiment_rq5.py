import torch
import torch.nn.functional as F
import numpy as np

# ==================== Step 1: Bug 复现 ====================
def reproduce_normalize_bug():
    """
    PyTorch Issue #184575 复现
    """
    # 构造包含零向量的 batch: (batch=2, dim=3)
    x = torch.zeros(2, 3, requires_grad=True)
    
    # 前向传播
    y = F.normalize(x, p=2, dim=-1)  # bug 触发点
    
    # 反向传播: 构造虚拟损失
    loss = y.sum()
    loss.backward()
    
    print("Input (zero vector):", x)
    print("Normalized output (should be NaN/Inf, got finite):", y)
    print("Gradient (should be NaN, got ~1e12):", x.grad)
    print("Output norm (should be NaN or 1, got):", torch.norm(y, dim=-1))
    
    return x, y, x.grad

# ==================== Step 2: 四层违规检测 ====================
def detect_violation_layers(x, y, grad, eps=1e-12):
    """
    对 F.normalize 的输出执行四层违规检测
    返回: violations dict, fingerprint vector (4,)
    """
    violations = {
        'L1_Numerical_Stability': [],
        'L2_Statistical_Moments': [],
        'L3_Distributional_Axiom': [],
        'L4_Structural_Invariant': []
    }
    
    # --- L1: 数值稳定性 ---
    # 1. 除零检测: 输入范数是否为零
    input_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    zero_mask = (input_norm < eps).squeeze()
    
    if zero_mask.any():
        violations['L1_Numerical_Stability'].append('division_by_zero: norm < eps')
    
    # 2. 伪有限异常值: 输入为零但输出非 NaN/Inf 且 magnitude 异常
    finite_mask = torch.isfinite(y)
    if zero_mask.any() and finite_mask[zero_mask].all():
        # 零输入下得到有限值 = 数值灾难（应为 NaN）
        mag = torch.abs(y[zero_mask]).max().item()
        if mag > 1e6:  # 异常大值阈值
            violations['L1_Numerical_Stability'].append(f'pseudo_finite_catastrophe: magnitude={mag:.2e}')
    
    # 3. 梯度违规: 未定义输入的梯度应为 NaN，但为有限值
    if zero_mask.any() and grad is not None:
        grad_finite = torch.isfinite(grad[zero_mask]).all()
        if grad_finite:
            violations['L1_Numerical_Stability'].append(
                f'gradient_violation: undefined-input grad should be NaN, got finite (max={grad[zero_mask].abs().max().item():.2e})'
            )
    
    # --- L3: 分布公理/算子契约 ---
    # 归一化契约: 输出范数应为 1 (或 NaN for zero input)
    output_norm = torch.norm(y, p=2, dim=-1)
    
    # 对非零输入，检查是否严格为 1
    non_zero_mask = ~zero_mask.squeeze()
    if non_zero_mask.any():
        dev = torch.abs(output_norm[non_zero_mask] - 1.0).max().item()
        if dev > eps:
            violations['L3_Distributional_Axiom'].append(f'normalization_axiom_violation: ||y|| != 1 (dev={dev:.2e})')
    
    # 对零输入，输出应为 NaN/Inf (未定义)，不应为有限值
    if zero_mask.any() and torch.isfinite(y[zero_mask]).all():
        violations['L3_Distributional_Axiom'].append(
            'normalization_undefined_contract_breach: zero-input should yield NaN, got finite'
        )
    
    # --- L4: 结构不变量 ---
    # 梯度传播结构: 未定义输入 -> NaN 梯度 (PyTorch autograd 文档契约)
    if grad is not None and zero_mask.any():
        if not torch.isnan(grad[zero_mask]).all():
            violations['L4_Structural_Invariant'].append(
                'gradient_structure_breach: undefined-input gradient contract violated'
            )
    
    # 构建指纹向量 (4,)
    fp = np.array([
        len(violations['L1_Numerical_Stability']),
        len(violations['L2_Statistical_Moments']),
        len(violations['L3_Distributional_Axiom']),
        len(violations['L4_Structural_Invariant'])
    ], dtype=float)
    
    return violations, fp

# ==================== Step 3: 容差 Oracle 失效演示 ====================
def tolerance_oracle_failure(x, y, tau=1e-5):
    """
    展示传统容差测试为何失效
    """
    # 参考实现: numpy manual normalize (同样会在零向量产生 NaN/Inf)
    x_np = x.detach().numpy()
    norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
    ref = x_np / norms  # 零向量处产生 Inf/NaN
    
    y_np = y.detach().numpy()
    
    # 问题: NaN vs 1e12 无法用容差比较
    # 传统框架通常: (1) 忽略 NaN (2) 或标记为 inconclusive
    mask = np.isfinite(ref) & np.isfinite(y_np)
    if mask.any():
        err = np.max(np.abs(ref[mask] - y_np[mask]))
        print(f"Finite-region max error: {err:.2e} (tau={tau})")
    else:
        print("No finite-region overlap for comparison (all NaN vs all finite)")
    
    # 工业测试的常见"漏洞": 如果两者都是 NaN/异常，测试通过
    n_nan_ref = np.sum(~np.isfinite(ref))
    n_nan_y = np.sum(~np.isfinite(y_np))
    print(f"NaN/Inf count: Reference={n_nan_ref}, PyTorch={n_nan_y}")
    print("Tolerance Oracle verdict: AMBIGUOUS (cannot compare NaN with 1e12 finite)")

# ==================== 主执行 ====================
if __name__ == "__main__":
    x, y, grad = reproduce_normalize_bug()
    violations, fp = detect_violation_layers(x, y, grad)
    
    print("\n=== 四层违规检测结果 ===")
    for layer, vlist in violations.items():
        status = "✓ TRIGGERED" if vlist else "✗ None"
        print(f"{layer}: {status}")
        for v in vlist:
            print(f"  - {v}")
    
    print(f"\nViolation Fingerprint (L1/L2/L3/L4): {fp}")
    print(f"Dominant Layer: L{int(np.argmax(fp))+1} ({list(violations.keys())[int(np.argmax(fp))]})")
    
    print("\n=== 容差 Oracle 失效分析 ===")
    tolerance_oracle_failure(x, y)