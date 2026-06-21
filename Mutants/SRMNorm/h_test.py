import torch
import numpy as np
import importlib
from Mutants.mutant_ananlysis import detect_fingerprint_collisions, analyze_single_operator,print_operator_summary

from Mutants.experiment_1 import plot_coverage,plot_cm_both,plot_km_fp_confusion_heatmap,plot_km_layer_heatmap

from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies,plot_cases
from Mutants.experiment_rq3 import run_rq3_experiment
from Mutants.experiment_rq4 import run_rq4_experiment_a,run_rq4_experiment_b

OPERATOR_MAPPING = {
        "M00": "ORIG", "M01": "AOR", "M02": "AOR", "M03": "AOR", "M04": "AOR",
        "M05": "AOR", "M06": "AOR", "M07": "AOR", "M08": "AOR", "M09": "AOR",
        "M10": "AOR", "M11": "AOR", "M12": "AOR", "M13": "ROR", "M14": "ROR",
        "M15": "ROR", "M16": "ROR", "M17": "ROR", "M18": "ROR", "M19": "ROR",
        "M20": "ROR", "M21": "CR", "M22": "CR", "M23": "CR", "M24": "CR",
        "M25": "CR", "M26": "CR", "M27": "CR", "M28": "CR", "M29": "CR",
        "M30": "CR", "M31": "SDL", "M32": "SDL", "M33": "SDL", "M34": "SDL",
        "M35": "SDL", "M36": "SDL", "M37": "VVR", "M38": "VVR", "M39": "VVR",
        "M40": "VVR", "M41": "FCR", "M42": "FCR", "M43": "FCR", "M44": "FCR",
        "M45": "FCR", "M46": "FCR", "M47": "FCR", "M48": "FCR", "M49": "FCR",
        "M50": "FCR", "M51": "ASR", "M52": "ASR", "M53": "ASR", "M54": "ASR",
        "M55": "BVR", "M56": "BVR", "M57": "BVR", "M58": "BVR", "M59": "BVR",
        "M60": "BVR", "M61": "SOR", "M62": "SOR", "M63": "SOR", "M64": "UOI",
        "M65": "UOI", "M66": "UOI", "M67": "UOI", "M68": "DTR", "M69": "DTR",
        "M70": "DTR", "M71": "EHR", "M72": "EHR", "M73": "RVR", "M74": "RVR",
        "M75": "RVR", "M76": "LVR", "M78": "CBR", "M79": "CBR", "M80": "STR",
        "M81": "STR", "M82": "IOR", "M83": "IOR", "M84": "IOR", "M85": "BOR",
        "M87": "BOR", "M90": "RMS", "M91": "RMS", "M92": "RMS", "M93": "RMS",
        "M94": "RMS", "M95": "RMS", "M96": "RMS", "M97": "RMS", "M98": "RMS",
        "M99": "RMS", "M100": "NPR", "M101": "NPR", "M102": "NPR", "M103": "MAR",
        "M104": "MAR", "M106": "DVR", "M107": "DVR", "M108": "BCR", "M109": "BCR",
        "M110": "GDR"
    }

# RMSNorm 行为违规类型定义 (基于原始论文与LLaMA工业实现)
all_behavior_types = [
    # === 数值稳定性类 (Critical for LLaMA) ===
    "nan_inf_output",           # 输出含NaN/Inf (除零或溢出)
    "zero_input_anomaly",       # 全零输入未返回零 (eps处理错误)
    "exploding_output",         # 输出值爆炸 (>1e6, 权重缩放错误)
    "vanishing_output",         # 输出值消失 (<1e-10, 逆向归一化)
    
    # === 统计特性类 (RMSNorm核心守恒) ===
    "mean_distortion",          # 均值被强制归零 (混淆为LayerNorm)
    "sign_flip",                # 符号翻转 (RMSNorm应保号)
    "variance_miscompute",      # 方差计算错误 (未用x^2或错误维度)
    "rms_deviation",            # RMS值偏离预期 (>5σ或<0.5σ)
    
    # === 参数违规类 (Weight/Epsilon) ===
    "bias_contamination",       # 错误引入bias项 (RMSNorm无bias)
    "eps_ineffective",          # epsilon未参与计算 (除零风险)
    "weight_shape_mismatch",    # 权重维度与hidden_size不符
    "uninitialized_weight",     # 权重未初始化或全零
    
    #=== 维度/形状类 ===
    "keepdim_violation",        # 维度压缩错误 (缺少keepdim=True)
    "broadcasting_failure",     # 广播机制失效 (weight*x_norm失败)
    
    # === 精度/性能类 (AI算子特有) ===
    "fp16_overflow",            # 半精度溢出 (LLaMA常见场景)
    # "inplace_modification",     # 输入张量被意外修改
    "gradient_anomaly",         # 反向传播梯度爆炸/消失
    
    # === 语义等价类 (等价变异体检测) ===
    #"algebraic_identity_break", # 代数恒等式被破坏 (如sqrt(x^2)!=|x|)
    "redundant_computation",    # 冗余计算导致数值漂移
]

# ==========================
# 2.载入 Oracle & Mutants
# ==========================
def load_oracle():
    try:
        spec = importlib.util.spec_from_file_location('M00', 'M00.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.RMSNorm
    except Exception as e:
        print(f"[WARN] Failed to load Oracle: {e}")
        return None
def load_mutants(mutant_files):
    mutant_funcs={}
    for mf in mutant_files:
        name=mf.split('.')[0]
        
        try:
            spec=importlib.util.spec_from_file_location(name,mf)
            mod=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mutant_funcs[name]=mod.RMSNorm
        except Exception as e:
            print(f'[WARM] 读取文件失败{mf}:{e}')
    return mutant_funcs
oracle = load_oracle()

import torch

def detect_rmsnorm_violations(
    output,
    x=None,
    weight=None,
    eps=1e-6,
    x_original=None,
    tol=1e-6,
    enable_gradient_check=False
):
    """
    Industrial RMSNorm violation detector (robust version)
    Supports half/float, broadcasting, shape mismatch handling
    """
    violations = []

    # --------------------------
    # 0. Basic checks
    # --------------------------
    if output is None or not isinstance(output, torch.Tensor):
        return ["invalid_output"]

    out = output
    x_safe = x.float() if x is not None else None
    out = out.float()
    weight_safe = weight.float() if weight is not None else None

    # --------------------------
    # 1. Numerical stability
    # --------------------------
    if torch.isnan(out).any() or torch.isinf(out).any():
        violations.append("nan_inf_output")

    out_max = out.abs().max().item()
    if out_max > 1e6:
        violations.append("exploding_output")
    if out_max < 1e-10 and x_safe is not None and x_safe.abs().max().item() > 1e-6:
        violations.append("vanishing_output")
    if x_safe is not None and torch.allclose(x_safe, torch.zeros_like(x_safe), atol=tol):
        if not torch.allclose(out, torch.zeros_like(out), atol=tol*10):
            violations.append("zero_input_anomaly")

    # --------------------------
    # 2. Statistical properties
    # --------------------------
    if x_safe is not None:
        x_mean = x_safe.mean().item()
        out_mean = out.mean().item()
        if abs(x_mean) > tol and abs(out_mean) < tol/10:
            violations.append("mean_distortion")

        # --- sign flip detection ---
        try:
            out_expanded = out
            if out.shape != x_safe.shape:
                out_expanded = out.expand_as(x_safe)
            mask = (x_safe.abs() > tol) & (out_expanded.abs() > tol)
            if mask.any():
                if not torch.equal(torch.sign(x_safe)[mask], torch.sign(out_expanded)[mask]):
                    violations.append("sign_flip")
        except Exception:
            # 广播失败，跳过符号检查
            pass

        # --- variance / RMS ---
        var_correct = (x_safe**2).mean(-1, keepdim=True)
        if weight_safe is not None:
            w = weight_safe
            if w.dim() == 1:
                w = w.view(1, -1)
            denom = (x_safe * w)
            denom[denom == 0] = 1e-8
            inferred = out_expanded / denom
            rms_est = inferred.mean(-1, keepdim=True)
            var_est = (1 / (rms_est**2)) - eps
            var_wrong = ((x_safe - x_safe.mean(-1, keepdim=True))**2).mean(-1, keepdim=True)
            if var_est.shape == var_wrong.shape and torch.allclose(var_est, var_wrong, atol=1e-4):
                violations.append("variance_miscompute")
            rms_true = torch.rsqrt(var_correct + eps)
            ratio = (rms_est / rms_true).mean().item()
            if ratio > 5 or ratio < 0.5:
                violations.append("rms_deviation")

        # --- algebraic identity / redundant computation ---
        if out.std().item() < 1e-8 and x_safe.std().item() > 1e-3:
            violations.append("redundant_computation")

    # --------------------------
    # 3. Parameter violations
    # --------------------------
    if weight_safe is not None and x_safe is not None:
        try:
            out_expanded = out
            if out.shape != x_safe.shape:
                out_expanded = out.expand_as(x_safe)
            mask = (x_safe.abs() > tol)
            if mask.any():
                ratios = (out_expanded[mask] / x_safe[mask]).detach().cpu().numpy()
                cv = ratios.std() / (abs(ratios.mean()) + 1e-10)
                if cv > 0.1:
                    violations.append("bias_contamination")
        except Exception:
            pass

    if weight_safe is not None:
        if torch.isnan(weight_safe).any():
            violations.append("uninitialized_weight")
        if torch.allclose(weight_safe, torch.zeros_like(weight_safe), atol=tol):
            violations.append("uninitialized_weight")
        if x_safe is not None and weight_safe.shape[0] != x_safe.shape[-1]:
            violations.append("weight_shape_mismatch")

    # --------------------------
    # 4. Shape violations
    # --------------------------
    if x_safe is not None:
        if x_safe.shape != out.shape:
            if len(x_safe.shape) != len(out.shape):
                violations.append("keepdim_violation")
            else:
                violations.append("broadcasting_failure")

    # --------------------------
    # 5. FP16 specific checks
    # --------------------------
    if x is not None and x.dtype == torch.float16:
        if (x.abs() > 256).any():
            violations.append("fp16_overflow")
    if x_original is not None and x_safe is not None:
        if not torch.allclose(x_original.float(), x_safe):
            violations.append("inplace_modification")

    # --------------------------
    # 6. Gradient anomaly
    # --------------------------
    if enable_gradient_check and x_safe is not None and x_safe.requires_grad:
        try:
            if x_safe.grad is not None:
                x_safe.grad.zero_()
            loss = out.sum()
            loss.backward(retain_graph=True)
            grad = x_safe.grad
            if grad is not None:
                gmax = grad.abs().max().item()
                gmin = grad.abs().min().item()
                if gmax > 1e6 or (0 < gmin < 1e-10):
                    violations.append("gradient_anomaly")
            x_safe.grad.zero_()
        except Exception:
            violations.append("gradient_anomaly")

    return sorted(list(set(violations)))

def layered_decision_engine(oracle_out, mutant_out, x, weight, eps=1e-6, 
                           x_original=None, enable_gradient_check=False, tol=1e-9):
    oracle_viol = detect_rmsnorm_violations(
        oracle_out, x=x, weight=weight, eps=eps, tol=tol,
        x_original=x_original,
        enable_gradient_check=enable_gradient_check  # 传递标志
    )
    
    mutant_viol = detect_rmsnorm_violations(
        mutant_out, x=x, weight=weight, eps=eps, tol=tol,
        x_original=x_original,
        enable_gradient_check=enable_gradient_check
    )
     
    
    # 转换为集合便于比较
    oracle_has = len(oracle_viol) > 0
    mutant_has = len(mutant_viol) > 0
    oracle_set = set(oracle_viol)
    mutant_set = set(mutant_viol)

    # Layer 3 判定逻辑
    
    # 情况1：两者都无违规 → 纯数值比对
    if not oracle_has and not mutant_has:
        is_killed = not np.allclose(oracle_out, mutant_out, atol=tol)
        return is_killed, oracle_viol, mutant_viol
    
    # 情况2：仅一方有违规（异或）→ 行为差异显著，杀死
    if oracle_has ^ mutant_has:
        return True, oracle_viol, mutant_viol
    
    # 情况3：都有违规，但类型不同 → 行为差异，杀死
    if oracle_set != mutant_set:
        return True, oracle_viol, mutant_viol
    
    # 情况4：都有违规且类型完全相同 → 可能等价（需进一步数值验证）
    # 即使违规类型相同，数值差异大也应杀死（同类型但程度不同）
    numerical_diff = not np.allclose(oracle_out, mutant_out, atol=tol)
    return numerical_diff, oracle_viol, mutant_viol

def run_test(func, x, eps=1e-6):

    global oracle

    # =====================================================
    # 1 Build models
    # =====================================================

    hidden_size = x.shape[-1]

    mutant = func(hidden_size, eps=eps)
    oracle_inst = oracle(hidden_size, eps=eps)

    weight = getattr(mutant, "weight", None)

    # =====================================================
    # 2 Input isolation (防止 inplace)
    # =====================================================

    if isinstance(x, torch.Tensor):

        x_original = x.detach().clone()

        x_mut = x.detach().clone()
        x_oracle = x.detach().clone()

        enable_grad = x.requires_grad

        if enable_grad:
            x_mut.requires_grad_(True)
            x_oracle.requires_grad_(True)

    else:

        x_original = None
        x_mut = x
        x_oracle = x
        enable_grad = False

    # =====================================================
    # 3 Run mutant
    # =====================================================

    mutant_out = None
    mutant_err = None

    try:

        if enable_grad:
            mutant_out = mutant(x_mut)
        else:
            with torch.no_grad():
                mutant_out = mutant(x_mut)

        if torch.isnan(mutant_out).any() or torch.isinf(mutant_out).any():
            mutant_err = "nan_inf_output"
            mutant_out = None

    except Exception:
        mutant_err = "exception"

    # =====================================================
    # 4 Run oracle
    # =====================================================

    oracle_out = None
    oracle_err = None

    try:

        if enable_grad:
            oracle_out = oracle_inst(x_oracle)
        else:
            with torch.no_grad():
                oracle_out = oracle_inst(x_oracle)

        if torch.isnan(oracle_out).any() or torch.isinf(oracle_out).any():
            oracle_err = "nan_inf_output"
            oracle_out = None

    except Exception:
        oracle_err = "exception"

    # =====================================================
    # 5 Exception comparison
    # =====================================================

    if mutant_err is not None or oracle_err is not None:

        if mutant_err == oracle_err:
            return False, [oracle_err], [mutant_err]

        return True, [oracle_err], [mutant_err]

    # =====================================================
    # 6 Violation detection
    # =====================================================

    oracle_viol = detect_rmsnorm_violations(
        oracle_out,
        x=x_oracle,
        weight=weight,
        eps=eps,
        x_original=x_original,
        enable_gradient_check=enable_grad
    )

    mutant_viol = detect_rmsnorm_violations(
        mutant_out,
        x=x_mut,
        weight=weight,
        eps=eps,
        x_original=x_original,
        enable_gradient_check=enable_grad
    )

    oracle_set = set(oracle_viol)
    mutant_set = set(mutant_viol)

    # =====================================================
    # 7 Behavior difference
    # =====================================================

    if oracle_set != mutant_set:
        return True, oracle_viol, mutant_viol

    # =====================================================
    # 8 Output comparison
    # =====================================================

    if isinstance(oracle_out, torch.Tensor) and isinstance(mutant_out, torch.Tensor):

        if oracle_out.shape != mutant_out.shape:
            return True, oracle_viol, mutant_viol

        try:

            if not torch.allclose(
                oracle_out.float(),
                mutant_out.float(),
                atol=1e-6
            ):
                return True, oracle_viol, mutant_viol

        except Exception:
            return True, oracle_viol, mutant_viol

    # =====================================================
    # 9 Equivalent
    # =====================================================

    return False, oracle_viol, mutant_viol

def build_fingerprints(mutant_funcs, tests):
    """
    构建RMSNorm变异体的行为指纹（仿照Softmax版本）
    
    Args:
        mutant_funcs: dict, {变异体名称: 模型实例(RMSNorm或其变异体)}
        tests: list, 测试用例列表。
               支持格式: [x_tensor, ...] 或 [(x_tensor, eps), ...]
               其中 x_tensor: [batch, seq_len, hidden_size]
    
    Returns:
        fingerprints: dict, {name: np.array([0,1,1,0...])} - 每个测试用例是否杀死该变异体
        ms_per_mutant: dict, {name: float} - 变异得分(杀死率)
        violation_map: dict, {name: np.array([count1, count2...])} - 20维行为违规计数
    """
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}
    
    # 假设 all_behavior_types 是全局定义的 RMSNorm 20项违规类型列表
    global all_behavior_types
    
    for name, func in mutant_funcs.items():
        # 20维行为向量（对应 all_behavior_types 的计数）
        vec = np.zeros(len(all_behavior_types))
        killed_list = []
        
        for test in tests:
            # 解包 test（适配 RMSNorm 的两种可能输入格式）
            if isinstance(test, tuple) and len(test) >= 1:
                x = test[0]                
            else:
                x = test
                # 从模型实例获取eps，否则默认1e-6
                eps = getattr(func, 'variance_epsilon', 1e-6)
            # eps 获取逻辑：如果 func 是类，用默认值；如果是实例，从实例属性获取
            if isinstance(func, type):
                eps = 1e-6  # 实例化时的默认 epsilon
            else:
                eps = getattr(func, 'variance_epsilon', 1e-6)
            # 调用 RMSNorm 版本的 run_test
            # 注意：run_test 内部会获取 func.weight，这里只需传递 x 和 eps
            killed, o_viol, m_viol = run_test(func, x, eps=eps)
            killed_list.append(1 if killed else 0)
            
            # 累加违规类型到20维向量（one-hot计数累加）
            for viol in m_viol:
                if viol in all_behavior_types:
                    idx = all_behavior_types.index(viol)
                    vec[idx] += 1
        
        # 计算该变异体的Mutation Score（杀死率）
        ms_per_mutant[name] = np.mean(killed_list) if killed_list else 0.0
        
        # fingerprints: kill矩阵的一行（每个测试用例是否杀死）
        fingerprints[name] = np.array(killed_list, dtype=float)
        
        # violation_map: 20维行为违规计数向量
        violation_map[name] = vec
    
    return fingerprints, ms_per_mutant, violation_map

def generate_lhs_samples(n_samples=50, hidden_size=4096, seq_len=8, batch_size=2, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # -------------------------------
    # 阶段1：边界注入 (占40%)
    # -------------------------------
    boundary_cases = [
        # 数值稳定性
        ("explode_to_inf", lambda: torch.full((batch_size, seq_len, hidden_size), 1e20)),
        ("div_by_zero_sim", lambda: torch.full((batch_size, seq_len, hidden_size), 0.0)),
        ("zero_input", lambda: torch.zeros(batch_size, seq_len, hidden_size)),
        # 精度 / 性能
        ("fp16_max", lambda: torch.full((batch_size, seq_len, hidden_size), 65504.0, dtype=torch.float16)),
        ("fp16_near_overflow", lambda: torch.full((batch_size, seq_len, hidden_size), 60000.0, dtype=torch.float16)),
        ("fp16_underflow", lambda: torch.full((batch_size, seq_len, hidden_size), 1e-8, dtype=torch.float16)),
        # 参数异常
        ("tiny_variance", lambda: torch.full((batch_size, seq_len, hidden_size), 1e-15)),
        ("near_eps", lambda: torch.full((batch_size, seq_len, hidden_size), 1e-7)),
        ("eps_boundary", lambda: torch.full((batch_size, seq_len, hidden_size), 1e-6)),
        # 梯度异常
        ("grad_explode", lambda: (torch.randn(batch_size, seq_len, hidden_size) * 1e5, True)),
        ("grad_vanish", lambda: (torch.randn(batch_size, seq_len, hidden_size) * 1e-10, True)),
        # 常规极值
        ("all_positive_small", lambda: torch.rand(batch_size, seq_len, hidden_size) * 1e-8),
        ("all_positive_large", lambda: torch.rand(batch_size, seq_len, hidden_size) * 1e5),
        ("all_negative_small", lambda: -torch.rand(batch_size, seq_len, hidden_size) * 1e-8),
        ("all_negative_large", lambda: -torch.rand(batch_size, seq_len, hidden_size) * 1e5),
        ("mixed_extreme", lambda: torch.randn(batch_size, seq_len, hidden_size) * 1e3),
        ("constant_positive", lambda: torch.ones(batch_size, seq_len, hidden_size) * 2.5),
        ("constant_negative", lambda: torch.ones(batch_size, seq_len, hidden_size) * (-2.5)),
    ]
    
    n_boundary = int(n_samples * 0.4)
    selected_boundaries = (boundary_cases * ((n_boundary // len(boundary_cases)) + 1))[:n_boundary]
    
    samples = []
    for desc, fn in selected_boundaries:
        result = fn()
        if isinstance(result, tuple):
            x, requires_grad = result
            x = x.clone().requires_grad_(requires_grad)
        else:
            x = result.clone()
        samples.append((x, {
            "type": "boundary",
            "desc": desc,
            "is_fp16": x.dtype == torch.float16,
            "requires_grad": x.requires_grad
        }))
    
    # -------------------------------
    # 阶段2：LHS随机采样 (剩余60%)
    # -------------------------------
    remaining = n_samples - len(samples)
    
    def lhs_sample(n, d):
        result = np.zeros((n, d))
        for i in range(d):
            intervals = np.linspace(0, 1, n + 1)
            points = intervals[:-1] + np.random.rand(n) / n
            np.random.shuffle(points)
            result[:, i] = points
        return result
    
    lhs_points = lhs_sample(remaining, 4)
    
    for i in range(remaining):
        p = lhs_points[i]
        mag = 10 ** (p[0] * 8 - 4)   # -4~4
        sign = 1 if p[1] > 0.5 else -1
        mean = (p[2] * 2 - 1) * 5
        
        x = torch.randn(batch_size, seq_len, hidden_size) * mag + mean
        x = x * sign
        
        # 避免极端爆炸
        x = torch.clamp(x, -1e7, 1e7)
        
        # 随机梯度检查
        if i % 2 == 0:
            x = x.clone().requires_grad_(True)
        
        samples.append((x, {
            "type": "lhs",
            "lhs_params": {
                "magnitude": float(x.abs().max()),
                "requires_grad": x.requires_grad
            },
            "is_fp16": False,
            "requires_grad": x.requires_grad
        }))
    
    return samples

# === 使用示例 ===
if __name__ == "__main__":
    categories = {
        'L1_Numerical_Stability':   [0, 1, 2, 3, 14, 15],   # nan/inf, 全零异常, 爆炸/消失, fp16溢出, 梯度异常
        'L2_Statistical_Properties': [4, 5, 6, 7],           # 均值失真, 符号翻转, 方差误算, RMS偏离
        'L3_Semantic_Logic':        [8, 9, 10, 11, 16],      # bias污染, eps失效, 权重维度/初始化, 冗余计算
        'L4_Structural_Dimension':  [12, 13],                # keepdim违规, 广播失败
    }
    samples = generate_lhs_samples(n_samples=50)
    mutant_files=[f"{k}.py" for k in OPERATOR_MAPPING.keys() if k!="M00"]    
    mutant_funcs=load_mutants(mutant_files)
    tests = generate_lhs_samples(n_samples=100)
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutant_funcs, tests)
    
#region experiment RQ1
    plot_cm_both(kill_matrix,violation_map)
    # plot_km_fp_confusion_heatmap(kill_matrix,violation_map)
    # matrix=plot_km_layer_heatmap(kill_matrix,violation_map,categories) #这个有问题
    # print(matrix)
#endregion
#region experiment RQ1
    # plot_km_fp_confusion_heatmap(kill_matrix,violation_map)
    # matrix=plot_km_layer_heatmap(kill_matrix,violation_map,categories) #这个有问题
    # print(matrix)
#endregion
    

    
#region RQ1 significance test
    # a=analyze_single_operator('SRMNorm',kill_matrix,violation_map)
    # print_operator_summary(a)
#endregion
    
#region RQ2:experiment A-C
    print('RQ2:experiment A: Metrics')
    v=compute_rq2_metrics(kill_matrix, violation_map,categories,n_fine=17)
    print(v)

    # print('RQ2:experiment B: fingerprint tsne')
    # plot_fingerprint_tsne(violation_map,categories,save_path='rq2\tsne.png')

    print('RQ2:experiment C: 3 Cases ')
    cases=extract_case_studies(kill_matrix,violation_map,categories,n_fine=17)
    for c in cases:
        print(f"\nCase: {c['m1']} vs {c['m2']}")
        print(f"  KM pattern: {c['km_pattern'][:5]}... (same class)")
        print(f"  FP({c['m1']}): {c['fp_m1']} → {c['layer_name_m1']} (L{c['dominant_m1']})")
        print(f"  FP({c['m2']}): {c['fp_m2']} → {c['layer_name_m2']} (L{c['dominant_m2']})")    
    plot_cases()
#endregion
 




    
