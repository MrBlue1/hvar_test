import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import importlib.util
from Mutants.experiment_1 import plot_cm_both,plot_coverage
from collections import Counter
from scipy import stats

from Mutants.mutant_ananlysis import detect_fingerprint_collisions, analyze_single_operator,print_operator_summary

from Mutants.experiment_1 import plot_coverage,plot_cm_both,plot_km_fp_confusion_heatmap,plot_km_layer_heatmap,plot_rq1_McNemar_test

from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies,plot_cases
from Mutants.experiment_rq3 import run_rq3_experiment,plot_HVAR_by_qr3_data
from Mutants.experiment_rq4 import run_rq4_experiment_a,run_rq4_experiment_b,experiment_qr4_plot

import pickle
import torch
import numpy as np

MUTANT_OPERATOR_MAP = {
    "M00": "ORIG",
    "M01": "ROR", "M02": "ROR", "M03": "ROR", "M04": "ROR",
    "M05": "ROR", "M06": "ROR", "M07": "ROR", "M08": "ROR",
    "M09": "ROR", "M10": "ROR", "M11": "ROR", "M12": "ROR",
    "M13": "AOR", "M14": "AOR", "M15": "AOR", "M16": "AOR",
    "M17": "AOR", "M18": "AOR", "M19": "AOR", "M20": "AOR",
    "M21": "AOR", "M22": "AOR",
    "M23": "LOR", "M24": "LOR", "M25": "LOR", "M26": "LOR",
    "M27": "LOR", "M28": "LOR", "M29": "LOR", "M30": "LOR",
    "M31": "COR", "M32": "COR", "M33": "COR", "M34": "COR",
    "M35": "COR", "M36": "COR",
    "M37": "UOI", "M38": "UOI", "M39": "UOI", "M40": "UOI",
    "M41": "UOI", "M42": "UOI", "M43": "UOI",
    "M44": "SDL", "M45": "SDL", "M46": "SDL", "M47": "SDL",
    "M48": "SDL", "M49": "SDL", "M50": "SDL",
    "M51": "ABS", "M52": "ABS", "M53": "ABS", "M54": "ABS",
    "M55": "ABS", "M56": "ABS", "M57": "ABS", "M58": "ABS",
    "M59": "ABS", "M60": "ABS", "M61": "SDL", "M62": "UOI",
    "M63": "SDL",
}

all_behavior_types = [    
    "invalid_output",      # 0: 非法输出/None
    "nan",                 # 1: 包含NaN
    "inf",                 # 2: 包含Inf
    "negative_probability", # 3: 负概率
    "exceeds_one",         # 4: 概率>1
    "numerical_underflow", # 5: 数值下溢(0)
    "probability_sum_violation",  # 6: 和不等于1
    "probability_contraction",    # 7: 和<1(收缩)
    "uniform_distribution",   # 8: 均匀分布(无区分度)
    "degenerate_distribution", # 9: 退化分布(独热)
    "low_entropy_sharp",      # 10: 过低熵(过于尖锐)
    "high_entropy_flat",      # 11: 过高熵(过于平坦)
    "dynamic_range_collapse", # 12: 动态范围崩溃
    "monotonicity_violation", # 13: 单调性破坏(logits与prob排序不一致)
    "topk_inconsistency",     # 14: Top-K不一致
    "broadcasting_error",     # 15: 广播错误(批次内全相同)
    "gradient_saturation",      # 16: 梯度饱和(输入差异过大)
    "gradient_explosion_risk",  # 17: 梯度爆炸风险(输入过于接近)
    "extreme_temperature",      # 18: 极端温度效应
    "cross_class_violation"     # 19: 跨类别归一化错误(错误axis)
]

#region 1.lhs测试数据生成
# ============================================
# 1.测试数据生成
# ============================================
def generate_lhs_samples(n: int, seed: Optional[int] = None, n_classes: int = 5, batch_size: int = 3):
    """分层LHS + 针对性违规触发器"""
    if seed:
        np.random.seed(seed)
    
    test_cases = []
    
    # 策略1: 标准LHS (40%) - 常规功能测试
    n_lhs = int(n * 0.4)
    # 扩展范围到极值
    dims = batch_size * n_classes
    strata = np.linspace(0, 1, n_lhs, endpoint=False)
    dimensions = []
    for i in range(dims):
        jitter = np.random.uniform(0, 1/n_lhs, n_lhs)
        dim_samples = strata + jitter
        np.random.shuffle(dim_samples)
        # 映射到 -100~100，但保留少量极端值
        values = np.where(
            np.random.rand(n_lhs) < 0.1,  # 10%极端值
            dim_samples * 2000 - 1000,     # [-1000, 1000]
            dim_samples * 200 - 100        # [-100, 100]
        )
        dimensions.append(values)
    
    for i in range(n_lhs):
        logits = np.array([d[i] for d in dimensions]).reshape(batch_size, n_classes)
        temp = np.random.choice([
            np.random.uniform(0.001, 0.1),  # 极低温度（尖锐分布）
            np.random.uniform(0.1, 5.0),    # 正常
            np.random.uniform(10, 100)      # 极高温度（平坦）
        ])
        axis = np.random.choice([-1, 0])
        # axis = -1 #不考虑错误维度
        test_cases.append((logits, temp, axis))
    
    # 策略2: 针对性故障触发 (40%) - 确保覆盖所有20种违规类型
    targeted_cases = [
        # 1. uniform_distribution: 所有logits相同 → 输出均匀
        (np.ones((batch_size, n_classes)) * 5.0, 1.0, -1),
        (np.zeros((batch_size, n_classes)), 10.0, -1),
        
        # 2. degenerate_distribution: 一个极大，其他极小 → 独热
        (np.array([[100, 0, 0, 0, 0], [0, 100, 0, 0, 0], [0, 0, 100, 0, 0]]), 0.01, -1),
        
        # 3. numerical_underflow: 极大负数经过exp → 0概率
        (np.full((batch_size, n_classes), -1000), 1.0, -1),
        (np.array([[-800, -750, -900, -850, -820]] * batch_size), 0.1, -1),
        
        # 4. dynamic_range_collapse: 混合极大极小，同时存在
        (np.array([[1000, -1000, 0, 0, 0], [500, -500, 100, -100, 0]]), 0.1, -1),
        
        # 5. broadcasting_error: 批次内完全相同（所有行一样）
        (np.array([[1, 2, 3, 4, 5]] * batch_size), 1.0, -1),
        
        # 6. gradient_saturation: 极大差异
        (np.array([[0, 0, 1000, 0, 0], [500, 500, 501, 500, 500]]), 0.01, -1),
        
        # 7. gradient_explosion_risk: 所有值极其接近（数值不稳定）
        (np.array([[1.0, 1.000001, 1.000002, 1.000003, 1.000004]] * batch_size), 1.0, -1),
        
        # 8. extreme_temperature: 极低温（T→0）和极高温（T→∞）
        (np.array([[1, 2, 3, 4, 5]]), 0.001, -1),   # 极低温
        (np.array([[1, 2, 3, 4, 5]]), 1000.0, -1),  # 极高温
        
        # 9. cross_class_violation: 错误axis测试（主要测试axis=0时的异常）
        (np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]).T, 1.0, 0),
        
        # 10. negative_probability / exceeds_one: 通过数值溢出产生（依赖变异体破坏）
        (np.array([[1e308, 1e308, 1e308]]), 1.0, -1),  # 可能导致inf
        
        # 11. probability_contraction/expansion: 非归一化输入+特定温度
        (np.array([[0.1, 0.1, 0.1, 0.1, 0.1]]), 0.01, -1),
        
        # 12. symmetry_violation等RBF相关（Softmax不适用，但保留数值异常）
        (np.array([[np.nan, 1, 2], [3, np.inf, 5], [6, 7, -np.inf]]), 1.0, -1),
        
        # 13. 混合极端：极小值与极大值共存
        (np.array([[1e-45, 1e308, 1, 1, 1]]), 1.0, -1),
        
        # 14. 空/退化维度测试
        (np.array([[]]), 1.0, -1),  # 空数组，测试invalid_output
         # nan: 输入包含nan 或 产生inf-inf
        (np.array([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]]), 1.0, -1),
        (np.array([[1e308, 1e308, 0]]), 1.0, -1),  # 极大值相减导致inf-inf
        
        # invalid_output: 空数组或错误维度
        (np.array([]).reshape(0, 5), 1.0, -1),  # 空批次
        (np.array([[]]), 1.0, -1),  # 空类别
        
        # negative_probability: 依赖变异体破坏，但需构造可产生负数的输入
        # 例如：如果变异体错误地使用了未shift的exp，极大负数可能产生负值（数值错误）
        (np.array([[-1000, -2000, -3000]]), 100.0, -1),  # 极高温+大负数
        
        # gradient_saturation: 输入差异 > 100
        (np.array([[0, 0, 0, 1000, 0]] * 3), 0.01, -1),
        
        # gradient_explosion_risk: 输入差异 < 1e-6（几乎相同）
        (np.array([[1.0, 1.0000001, 1.0000002]] * 3), 1.0, -1),
        
        # exceeds_one: 通常需要数值精度错误，构造极大值使求和约等于1但个别值>1
        (np.array([[1e16, 1e16]]), 1e-10, -1),  # 极低温+极大值
        # 在 generate_lhs_samples 的 targeted_cases 中增加：
        ( np.array([[1e200, 1e-200, 1e200, 1e-200]]), 0.01, -1 ),  # 极大极小交替
        ( np.array([[1e308, 1.0, 1e-308]]), 1.0, -1 ),              # 浮点极限值
        # 专门触发 exceeds_one (M63)：中等正数，差异明显
        # 产生 exp(2)≈7.39, exp(1.5)≈4.48, exp(3)≈20.08 等 >1 的值
        (np.array([[2.0, 0.0, 0.0]]), 1.0, -1),      # exp(2)=7.39 > 1
        (np.array([[1.5, 0.5, 0.0]]), 1.0, -1),      # exp(1.5)=4.48 > 1
        (np.array([[3.0, 1.0, 0.0]]), 1.0, -1),      # exp(3)=20.08 > 1
        (np.array([[0.7, 0.0, 0.0]]), 1.0, -1),      # exp(0.7)=2.01 > 1（临界测试）
        
        # 多批次测试（确保不同样本都触发）
        (np.array([[2.0, 0.0], [1.0, 0.0], [3.0, 1.0]]), 1.0, -1),
        
        # 不同温度测试（温度<1会放大差异，但可能溢出；温度>1会缩小，可能<1）
        (np.array([[2.0, 0.0, 0.0]]), 0.5, -1),      # 放大：exp(4)=54.6（检查是否inf）
        (np.array([[2.0, 0.0, 0.0]]), 2.0, -1),      # 缩小：exp(1)=2.718 > 1（安全）
    ]
    
    # 重复填充到40%配额
    n_targeted = int(n * 0.4)
    while len(targeted_cases) < n_targeted:
        # 对已有用例添加微小扰动，生成变体
        base = targeted_cases[len(targeted_cases) % len(targeted_cases)][0].copy()
        # base = targeted_cases[np.random.randint(len(targeted_cases))][0].copy() #防止每次都选第一个用例
        noise = np.random.randn(*base.shape) * 0.01
        targeted_cases.append((base + noise, 
                              np.random.uniform(0.5, 2.0), 
                              np.random.choice([-1, 0])))
    
    test_cases.extend(targeted_cases[:n_targeted])
    
    # 策略3: 边界攻击 (20%) - 专门测试数值边界
    n_boundary = n - len(test_cases)
    boundary_cases = []
    
    # 生成接近数值极限的用例
    for _ in range(n_boundary):
        # 随机选择攻击模式
        mode = np.random.choice([
            'overflow',      # 极大值
            'underflow',     # 极小值
            'precision',     # 精度极限
            'mixed_scale'    # 混合尺度
        ])
        
        if mode == 'overflow':
            scale = np.random.uniform(1e100, 1e308)
            logits = np.random.randn(batch_size, n_classes) * scale
            temp = np.random.uniform(0.1, 1.0)
        elif mode == 'underflow':
            scale = np.random.uniform(1e-300, 1e-100)
            logits = np.random.randn(batch_size, n_classes) * scale
            temp = np.random.uniform(0.1, 10.0)
        elif mode == 'precision':
            # 非常接近的数值，测试浮点精度
            base = np.random.uniform(-10, 10)
            logits = base + np.random.randn(batch_size, n_classes) * 1e-15
            temp = 1.0
        else:  # mixed_scale
            logits = np.random.choice(
                [1e-300, 1e-10, 0.1, 1.0, 10.0, 1e10, 1e100], 
                size=(batch_size, n_classes)
            )
            temp = np.random.uniform(0.01, 10)
            
        axis = np.random.choice([-1, 0])
        boundary_cases.append((logits, temp, axis))
    
    test_cases.extend(boundary_cases)
    
    # 打乱顺序，避免同类型聚集
    np.random.shuffle(test_cases)
    
    return test_cases

def generate_and_save_test_suite(n=100, seed=42, save_path='fixed_test_suite.pkl'):
    """生成LHS测试套件并序列化保存，确保永久固定"""
    # 固定随机种子（确保可重复）
    np.random.seed(seed)
    
    # 生成测试套件
    test_cases = generate_lhs_samples(
        n=n, 
        seed=seed,  # 内部也设置seed
        n_classes=5, 
        batch_size=3
    )
    
    # 保存到文件，后续所有实验都用这份
    with open(save_path, 'wb') as f:
        pickle.dump({
            'test_cases': test_cases,
            'seed': seed,
            'n': n,
            'metadata': 'Fixed LHS suite for RQ4 experiments'
        }, f)
    
    print(f"测试套件已固定保存至 {save_path}，包含 {len(test_cases)} 个用例")
    return test_cases

def load_fixed_test_suite(load_path='fixed_test_suite.pkl'):
    """加载固定的LHS测试套件，确保VP、KG等方法使用完全相同输入"""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loading Fixed Tests（seed={data['seed']}, n={len(data['test_cases'])}）")
    return data['test_cases']

#endregion

# ==========================
# 2.载入 Oracle & Mutants
# ==========================
def load_oracle():
    try:
        spec = importlib.util.spec_from_file_location('M00', 'M00.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.stable_softmax
    except Exception as e:
        print(f"[WARN] Failed to load Oracle: {e}")
        return None

def load_mutants(mutant_files):
    mutant_funcs = {}
    for mf in mutant_files:
        name = mf.split(".")[0]
        try:
            spec = importlib.util.spec_from_file_location(name, mf)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mutant_funcs[name] = mod.stable_softmax #定义运行的函数的名称：stable_softmax
        except Exception as e:
            print(f"[WARN] Failed to load {mf}: {e}")
    return mutant_funcs

# ==========================
# 3.三层决策引擎
# ==========================
def detect_generic_violations(output):
    """通用异常检测（保持兼容性）"""
    violations = []
    if np.isnan(output).any():
        violations.append("nan")
    if np.isinf(output).any():
        violations.append("inf")
    return violations

def detect_softmax_violations(S, logits=None, axis=-1, tol=1e-7):
    """
    检测Softmax输出的数学性质违规。
    
    Args:
        S: Softmax输出（概率分布）
        logits: 原始输入logits（可选，用于检查单调性保持）
        axis: 归一化轴（默认-1）
        tol: 数值容差
    
    Returns:
        list: 检测到的违规类型标签
    """
    violations = []

    # 严格检查 invalid_output（覆盖 index 0）
    if S is None:
        return ["invalid_output"]
    if not isinstance(S, np.ndarray):
        return ["invalid_output"]
    if S.size == 0 or S.shape[-1] == 0:  # 空数组或最后一维为0
        return ["invalid_output"]
    
    # 显式检查 nan（覆盖 index 1）
    if np.isnan(S).any():
        violations.append("nan")
    
    # 1. 基础数值异常（继承通用检测）
    if S is None or not isinstance(S, np.ndarray):
        return ["invalid_output"]
    
    if np.isnan(S).any():
        violations.append("nan")
    if np.isinf(S).any():
        violations.append("inf")
    
    # 维度检查
    if S.ndim < 1 or S.ndim > 3:
        violations.append("invalid_dimensions")
        return violations
    
    # 2. 概率基本公理检查
    # 非负性（Softmax输出理论上必>0，但数值下溢可能产生0）
    if (S < -tol).any():
        violations.append("negative_probability")
    if (S > 1.0 + tol).any():
        violations.append("exceeds_one")
    
    # 数值下溢（出现0概率，破坏对数空间计算）
    if (S < tol).any():
        violations.append("numerical_underflow")
    
    # 3. 归一化约束（和为1）
    prob_sums = S.sum(axis=axis)
    if not np.allclose(prob_sums, 1.0, atol=tol):
        violations.append("probability_sum_violation")
        # 检查是收缩还是扩张
        if np.any(prob_sums < 1.0 - tol):
            violations.append("probability_contraction")
        if np.any(prob_sums > 1.0 + tol):
            violations.append("probability_expansion")
    
    # 4. 分布形态异常
    # 沿指定轴分析
    n_classes = S.shape[axis] if axis < S.ndim else 1
    
    # 熵计算（检测极端分布）
    # 使用clip避免log(0)
    S_clipped = np.clip(S, 1e-300, 1.0)
    entropy = -np.sum(S_clipped * np.log(S_clipped), axis=axis)
    max_entropy = np.log(n_classes)
    
    # 均匀分布检查（无区分度，可能是变异体破坏了竞争性）
    if np.allclose(entropy, max_entropy, atol=0.01):
        violations.append("uniform_distribution")
    
    # 退化分布检查（独热近似，数值不稳定）
    max_probs = S.max(axis=axis)
    if np.all(max_probs > 1.0 - tol):
        violations.append("degenerate_distribution")
    
    # 低熵检查（过于自信，可能是温度参数变异）
    if np.all(entropy < max_entropy * 0.1):
        violations.append("low_entropy_sharp")
    
    # 高熵检查（过于模糊）
    if np.all(entropy > max_entropy * 0.95) and n_classes > 2:
        violations.append("high_entropy_flat")
    
    # 5. 数值精度与动态范围
    # 检查同时存在极大和极小值（动态范围崩溃）
    if S.size > 0:
        max_val = S.max()
        min_val_nonzero = S[S > 0].min() if np.any(S > 0) else 1.0
        
        # 方法1：使用对数比较（推荐，数值稳定）
        if min_val_nonzero > 0 and max_val > 0:
            log_ratio = np.log(max_val) - np.log(min_val_nonzero)
            if log_ratio > 460.5:  # ln(1e200) ≈ 460.5
                violations.append("dynamic_range_collapse")
    
    # 6. 单调性保持检查（如果提供logits）
    if logits is not None and logits.shape == S.shape:
        # 检查排序一致性：logits大的，softmax值也应该大
        logit_order = np.argsort(logits, axis=axis)
        softmax_order = np.argsort(S, axis=axis)
        if not np.allclose(logit_order, softmax_order):
            violations.append("monotonicity_violation")
        
        # Top-k一致性检查
        k = min(3, n_classes)
        logit_topk = np.argsort(logits, axis=axis)[..., -k:]
        softmax_topk = np.argsort(S, axis=axis)[..., -k:]
        if not np.allclose(logit_topk, softmax_topk):
            violations.append("topk_inconsistency")
    
    # 7. 维度特定检查
    if S.ndim == 2:
        # 批处理一致性：批次内分布不应完全相同
        if S.shape[0] > 1:
            first_dist = S[0]
            if np.allclose(S, first_dist):
                violations.append("broadcasting_error")
        
        # 修复：cross_class_violation 仅当归一化轴错误时检测
        actual_axis = axis if axis >= 0 else S.ndim + axis
        # Softmax 正确轴应为最后一维（axis=-1 等价于 axis=ndim-1）
        if actual_axis != S.ndim - 1:
            other_axis = 1 - actual_axis  # 对于2D，得到另一个轴的索引
            cross_sums = S.sum(axis=other_axis)
            if np.any(cross_sums < 0.9) or np.any(cross_sums > 1.1):
                violations.append("cross_class_violation")
    
    # 8. 温度参数异常（数值扩散）
    # 通过统计分布的方差推断温度效应
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            prob_variance = S.var(axis=axis).mean()
            if prob_variance < 1e-8:
                violations.append("extreme_temperature")
        except:
            pass  # 计算失败时静默处理
    
    # 9. 梯度敏感区域检测（数值不稳定点）
    # Softmax在输入差异大时梯度消失，在输入相等时梯度最大
    if logits is not None:
        # 使用 errstate 忽略溢出，因为 inf > 100 正是我们要检测的饱和情况
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                # 计算与均值的偏差，处理极端数值
                mean_logits = np.mean(logits, axis=axis, keepdims=True)
                diff = np.abs(logits - mean_logits)
                max_logit_diff = np.max(diff, axis=axis)
                
                if np.any(max_logit_diff > 100):  # 极大logit差异导致梯度消失
                    violations.append("gradient_saturation")
                if np.all(max_logit_diff < 1e-6):  # 输入几乎相同，数值不稳定
                    violations.append("gradient_explosion_risk")
            except:
                # 如果计算完全失败（如全为nan），标记为数值异常
                violations.append("numerical_error")
    
    return sorted(list(set(violations)))

# ==========================
# 4.运行测试
# ==========================
oracle = load_oracle()
def layered_decision_engine(oracle_out, mutant_out, logits=None, axis=-1):
    tol = 1e-9
    # 传递 logits 参数
    oracle_viol = detect_softmax_violations(oracle_out, logits=logits, axis=axis, tol=tol)
    mutant_viol = detect_softmax_violations(mutant_out, logits=logits, axis=axis, tol=tol)

    oracle_has = len(oracle_viol) > 0
    mutant_has = len(mutant_viol) > 0    

    # if not oracle_has and not mutant_has:
    #     return not np.allclose(oracle_out, mutant_out, atol=tol), oracle_viol, mutant_viol
    # if oracle_has ^ mutant_has:
    #     return True, oracle_viol, mutant_viol
    # if set(oracle_viol) != set(mutant_viol):
    #     return True, oracle_viol, mutant_viol
    # return False, oracle_viol, mutant_viol
    return not np.allclose(oracle_out, mutant_out, atol=tol), oracle_viol, mutant_viol

# 5. 修改 run_test，接收 axis 并传递
def run_test(func, logits, temperature=1.0, axis=-1):
    old = np.seterr(over='raise', invalid='raise')
    S, S_O = None, None
    f_err, o_err = None, None
    
    try:
        try:
            S = func(logits, temperature=temperature)
        except FloatingPointError:
            f_err = "inf"
        except Exception:
            f_err = "exception"
        
        if S is None:
            S = []
            f_err = "exception" if f_err is None else f_err
        
        try:
            S_O = oracle(logits, temperature=temperature)
        except FloatingPointError:
            o_err = "inf"
        except Exception:
            o_err = "exception"
            
        if S_O is None:
            S_O = []
            o_err = "exception" if o_err is None else o_err
        
        if f_err is not None and o_err is not None:
            return f_err != o_err, [o_err], [f_err]  # 修复：不同错误类型才算杀死
        if f_err:
            return True, [], [f_err]
        if o_err:
            return True, [o_err], []
        
        # 传递 axis 参数
        killed, o_viol, m_viol = layered_decision_engine(S_O, S, logits=logits, axis=axis)
        return killed, o_viol, m_viol
    finally:
        np.seterr(**old)

# ==========================
# 6.构建指纹 & MS & 违规类型覆盖
# ==========================
def build_fingerprints(mutant_funcs, tests):
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}
    all_viol_types = set()
    
    for name, func in mutant_funcs.items():
        vec = np.zeros(len(all_behavior_types))
        killed_list = []
        
        for i, test in enumerate(tests):
            # 解包 test
            if isinstance(test, tuple) and len(test) == 3:
                logits, temp, axis = test
            else:
                logits, temp, axis = test, 1.0, -1
            
            # 传递 axis 给 run_test
            killed, o_viol, m_viol = run_test(func, logits, temperature=temp, axis=axis)
            killed_list.append(1 if killed else 0)
            
            for viol in m_viol:
                if viol in all_behavior_types:
                    idx = all_behavior_types.index(viol)
                    vec[idx] += 1

        ms_per_mutant[name] = np.mean(killed_list) if killed_list else 0.0
        fingerprints[name] = np.array(killed_list, dtype=float)
        violation_map[name] = vec
        
    return fingerprints, ms_per_mutant, violation_map

def save_fingerprints(k_m,ms_per_mutant,violation_map,save_path='vp.pkl'):
    with open(save_path, 'wb') as f:
        pickle.dump({
            'violation_map': violation_map,
            'kill_matrix': k_m,
            'ms_per_mutant': ms_per_mutant,            
        }, f)
    
    print(f"k_m, ms_per_mutant, violation_map已固定保存至 {save_path}")
def load_fingerprints(load_path='vp.pkl'):
    with open(load_path, 'rb') as f:
        fingerprints=pickle.load(f)
    return fingerprints


#region 数值扰动试验
# ============================================
# 实验二（修正版）：输入空间扰动的数值饱和吸收实验
# 变更点：增加细粒度 20 种违规类型的独立统计与穿透分析
# ============================================
# -------------------------------------------------
# 0. 四层分类映射（与原有代码严格一致）
# -------------------------------------------------
LAYER_MAP = {
    'L1 Num.Stab':   [0, 1, 2, 16, 17, 18],
    'L2 Stat.Mom': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'L3 Sem.Logic':        [13, 14],
    'L4 Struct.Inv':  [15, 19],
}
LAYER_NAMES = list(LAYER_MAP.keys())

def generate_lhs_samples_v2(n: int, seed: Optional[int] = None, 
                            n_classes: int = 5, batch_size: int = 3):
    if seed:
        np.random.seed(seed)
    
    test_cases = []
    
    # 1. 标准 LHS (70%) —— [-10, 10]
    n_std = int(n * 0.70)
    dims = batch_size * n_classes
    strata = np.linspace(0, 1, n_std, endpoint=False)
    dimensions = []
    for i in range(dims):
        jitter = np.random.uniform(0, 1/n_std, n_std)
        dim_samples = strata + jitter
        np.random.shuffle(dim_samples)
        values = dim_samples * 20 - 10
        dimensions.append(values)
    
    for i in range(n_std):
        logits = np.array([d[i] for d in dimensions]).reshape(batch_size, n_classes)
        temp = np.random.uniform(0.5, 2.0)
        axis = -1
        test_cases.append((logits, temp, axis))
    
    # 2. 轻度边界 (20%)
    n_mild = int(n * 0.20)
    for _ in range(n_mild):
        mode = np.random.choice(['moderate_extreme', 'mixed_scale', 'near_boundary'])
        if mode == 'moderate_extreme':
            logits = np.random.randn(batch_size, n_classes) * np.random.uniform(100, 1000)
        elif mode == 'mixed_scale':
            logits = np.random.randn(batch_size, n_classes)
            logits[0] *= 100
            logits[1] *= 0.01
        else:
            logits = np.random.uniform(1e30, 1e35, size=(batch_size, n_classes))
            logits *= np.random.choice([-1, 1], size=(batch_size, n_classes))
        
        temp = np.random.uniform(0.1, 5.0)
        axis = -1
        test_cases.append((logits, temp, axis))
    
    # 3. 针对性故障 (10%) —— 显式用 float，避免 int/float 冲突
    n_target = n - len(test_cases)
    targeted_pool = [
        (np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [5.0, 4.0, 3.0, 2.0, 1.0],
                   [1.0, 3.0, 2.0, 4.0, 5.0]], dtype=np.float64), 1.0, -1),
        
        (np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], 
                   [4., 4., 4.], [5., 5., 5.]], dtype=np.float64).T, 1.0, 0),
        
        (np.full((batch_size, n_classes), -80.0, dtype=np.float64), 0.5, -1),
        
        (np.array([[-50., -60., -70., -80., -90.]] * batch_size, dtype=np.float64), 1.0, -1),
        
        (np.array([[10., 0., 0., 0., 0.],
                   [0., 10., 0., 0., 0.],
                   [0., 0., 10., 0., 0.]], dtype=np.float64), 0.1, -1),
    ]
    
    for i in range(n_target):
        base = targeted_pool[i % len(targeted_pool)]  # ← 补上这行
        # 关键修复：copy() 后显式转 float64，彻底隔离 int 污染
        logits = base[0].copy().astype(np.float64)
        if np.random.rand() < 0.5:
            logits += np.random.randn(*logits.shape) * 0.1
        test_cases.append((logits, base[1], base[2]))
    
    np.random.shuffle(test_cases)
    return test_cases

# -------------------------------------------------
# 1. 扰动施加函数（与上一版相同）
# -------------------------------------------------
def perturb_gaussian_noise(logits: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.randn(*logits.shape).astype(logits.dtype) * sigma
    return logits + noise

def perturb_extreme_boundary(logits: np.ndarray, p: float = 0.05) -> np.ndarray:
    """
    极端边界值：将 p% 元素替换为数值极限、Inf、NaN 或零。
    修复：使用 Python float 字面量 + float64 构造，避免 np.finfo 标量触发 OverflowError。
    """
    perturbed = logits.copy()
    mask = np.random.rand(*logits.shape) < p
    if not np.any(mask):
        return perturbed
    
    # 显式用 float64 构造，彻底绕开 np.finfo 标量的类型歧义
    choices = np.array([
        3.4e38,          # 极大正值（> float32 max，会被 clamp 或变成 inf）
        -3.4e38,         # 极大负值
        1.2e-38,         # 极小正值（接近 float32 最小正规数）
        np.inf, -np.inf,
        0.0, -0.0,
        np.nan
    ], dtype=np.float64)
    
    extreme_vals = np.random.choice(choices, size=np.count_nonzero(mask))
    perturbed[mask] = extreme_vals.astype(logits.dtype)
    return perturbed

def perturb_adversarial_shift(logits: np.ndarray, delta: float = 0.01) -> np.ndarray:
    direction = np.sign(np.random.randn(*logits.shape)).astype(logits.dtype)
    return logits + delta * direction

def perturb_precision_degradation(logits: np.ndarray) -> np.ndarray:
    return logits.astype(np.float16).astype(np.float32)

PERTURBATION_REGISTRY = {
    'clean':                lambda logits, **kw: logits,
    'gaussian_noise':       perturb_gaussian_noise,
    'extreme_boundary':     perturb_extreme_boundary,
    'adversarial_shift':    perturb_adversarial_shift,
    'precision_degradation': perturb_precision_degradation,
}

def apply_perturbation(test_case: Tuple, perturb_type: str,
                       perturb_params: Optional[dict] = None) -> Tuple:
    logits, temp, axis = test_case
    logits = np.array(logits, copy=True)
    params = perturb_params or {}
    perturbed_logits = PERTURBATION_REGISTRY[perturb_type](logits, **params)
    return (perturbed_logits, temp, axis)

# -------------------------------------------------
# 2. 运行 Oracle 并返回细粒度违规标签
# -------------------------------------------------
def run_oracle_violation(oracle_func, test_case: Tuple) -> List[str]:
    logits, temp, axis = test_case
    try:
        output = oracle_func(logits, temperature=temp)
        violations = detect_softmax_violations(output, logits=logits, axis=axis)
    except FloatingPointError:
        violations = ['inf']
    except Exception:
        violations = ['invalid_output']
    return violations

# -------------------------------------------------
# 3. 细粒度 -> 四层聚合（两个通道同时保留）
# -------------------------------------------------
def aggregate_violation_layers(violations: List[str]) -> Dict[str, int]:
    triggered = {layer: 0 for layer in LAYER_NAMES}
    for v in violations:
        if v not in all_behavior_types:
            continue
        idx = all_behavior_types.index(v)
        for layer_name, indices in LAYER_MAP.items():
            if idx in indices:
                triggered[layer_name] = 1
                break
    return triggered

def build_fine_grained_counter(violation_records: List[List[str]]) -> Dict[str, int]:
    """
    统计 20 种细粒度违规各自的触发次数。
    返回: {viol_type: count}
    """
    flat = [v for case_viols in violation_records for v in case_viols]
    return dict(Counter(flat))

# -------------------------------------------------
# 4. 核心实验引擎（同时记录细粒度 + 四层）
# -------------------------------------------------
def run_saturation_experiment(test_cases: List[Tuple],
                              oracle_func,
                              perturbation_configs: Dict[str, dict]) -> Dict:
    """
    返回结构:
    {
      pert_name: {
        'violation_records': List[List[str]],   # 每条用例的细粒度违规标签
        'fine_grained_vtr': Dict[str, float],   # 20 种违规各自的触发率
        'layer_matrix': np.ndarray,              # (n_cases, 4)  0/1
        'vtr': Dict[str, float],                 # 四层触发率
      }
    }
    """
    results = {}
    n_cases = len(test_cases)
    
    for pert_name, params in perturbation_configs.items():
        print(f"[实验二] 正在运行扰动: {pert_name} ...")
        violation_records = []
        layer_matrix = np.zeros((n_cases, 4), dtype=int)
        
        for i, case in enumerate(test_cases):
            perturbed_case = apply_perturbation(case, pert_name, params)
            viols = run_oracle_violation(oracle_func, perturbed_case)
            violation_records.append(viols)
            
            layer_trig = aggregate_violation_layers(viols)
            for j, layer_name in enumerate(LAYER_NAMES):
                layer_matrix[i, j] = layer_trig[layer_name]
        
        # 细粒度 VTR（20 种）
        fg_counter = build_fine_grained_counter(violation_records)
        fine_grained_vtr = {
            vtype: fg_counter.get(vtype, 0) / n_cases 
            for vtype in all_behavior_types
        }
        
        # 四层 VTR
        vtr = {
            layer: layer_matrix[:, j].sum() / n_cases
            for j, layer in enumerate(LAYER_NAMES)
        }
        
        results[pert_name] = {
            'violation_records': violation_records,
            'fine_grained_vtr': fine_grained_vtr,
            'layer_matrix': layer_matrix,
            'vtr': vtr,
            'total_cases': n_cases
        }
    
    return results

# -------------------------------------------------
# 5. 统计检验（四层层面，与上一版相同）
# -------------------------------------------------
def paired_saturation_test(clean_result: dict, pert_result: dict) -> Dict[str, dict]:
    clean_mat = clean_result['layer_matrix']
    pert_mat  = pert_result['layer_matrix']
    assert clean_mat.shape == pert_mat.shape
    
    test_report = {}
    for j, layer in enumerate(LAYER_NAMES):
        c, p = clean_mat[:, j], pert_mat[:, j]
        try:
            # 修正：stats.wilcoxon（不是 stat.wilcoxon）
            stat, pvalue = stats.wilcoxon(c, p, zero_method='zsplit')
        except ValueError:
            # 若所有差值为0，wilcoxon 可能报错
            stat, pvalue = 0.0, 1.0
        
        diff = p.astype(float) - c.astype(float)
        mean_diff, std_diff = np.mean(diff), np.std(diff, ddof=1)
        cohen_d = mean_diff / (std_diff + 1e-12) if std_diff > 0 else 0.0
        absorbed = (pvalue > 0.05) and (abs(cohen_d) < 0.2)
        
        test_report[layer] = {
            'clean_vtr':   float(np.mean(c)),
            'pert_vtr':    float(np.mean(p)),
            'delta_vtr':   float(np.mean(p)) - float(np.mean(c)),
            'wilcoxon_p':  float(pvalue),
            'cohens_d':    float(cohen_d),
            'absorbed':    bool(absorbed),
            'n_cases':     len(c)
        }
    return test_report
# -------------------------------------------------
# 6. 报表输出（新增细粒度分布表 + 穿透分析）
# -------------------------------------------------
def print_saturation_report(results: dict,
                            stat_reports: Optional[Dict[str, dict]] = None):
    print("\n" + "=" * 80)
    print("实验二：输入空间扰动的数值饱和吸收实验报告")
    print("=" * 80)
    
    # ---------- 6.1 细粒度违规触发率表（20 种） ----------
    print("\n【细粒度违规触发率 VTR (Top 触发项)】")
    print(f"{'扰动策略':<<18}" + "".join([f"{v:<<10}" for v in all_behavior_types[:10]]))
    print("-" * (18 + 10 * 10))
    
    # 打印前 10 种（可展开为全部 20 种）
    for pert_name, data in results.items():
        line = f"{pert_name:<18}"
        for vtype in all_behavior_types[:10]:
            line += f"{data['fine_grained_vtr'][vtype]:<<10.4f}"
        print(line)
    
    # 若需完整 20 种，可取消下面注释：
    # for pert_name, data in results.items():
    #     print(f"\n--- {pert_name} 全部 20 种违规 VTR ---")
    #     for vtype, rate in data['fine_grained_vtr'].items():
    #         if rate > 0:
    #             print(f"  {vtype:<30}: {rate:.4f}")
    
    # ---------- 6.2 四层聚合 VTR 总表 ----------
    print("\n【四层违规触发率 VTR】")
    header = f"{'扰动策略':<<20}" + "".join([f"{L:<24}" for L in LAYER_NAMES])
    print(header)
    print("-" * len(header))
    for pert_name, data in results.items():
        line = f"{pert_name:<20}"
        for layer in LAYER_NAMES:
            line += f"{data['vtr'][layer]:<<24.4f}"
        print(line)
    
    # ---------- 6.3 统计检验 ----------
    if stat_reports:
        print("\n【配对统计检验 (Wilcoxon + Cohen's d)】")
        for pert_name, report in stat_reports.items():
            print(f"\n  > 扰动: {pert_name}")
            for layer in LAYER_NAMES:
                r = report[layer]
                status = "【饱和吸收】" if r['absorbed'] else "【显著穿透】"
                print(f"    {layer:<26}: ΔVTR={r['delta_vtr']:+.4f}, "
                      f"d={r['cohens_d']:+.4f}, p={r['wilcoxon_p']:.4f}  {status}")
    
    # ---------- 6.4 穿透层级分析（新增） ----------
    print("\n【违规穿透层级分析】")
    print("说明：统计每种扰动下，20种细粒度违规分别落入哪一层；")
    print("      若某层在扰动后仍无任何细粒度违规被触发，则该层压力被完全吸收。\n")
    
    for pert_name, data in results.items():
        if pert_name == 'clean':
            continue
        fg_vtr = data['fine_grained_vtr']
        layer_active = {layer: [] for layer in LAYER_NAMES}
        
        for vtype, rate in fg_vtr.items():
            if rate == 0 or vtype not in all_behavior_types:
                continue
            idx = all_behavior_types.index(vtype)
            for layer_name, indices in LAYER_MAP.items():
                if idx in indices:
                    layer_active[layer_name].append((vtype, rate))
                    break
        
        print(f"  > {pert_name}:")
        for layer in LAYER_NAMES:
            items = layer_active[layer]
            if not items:
                print(f"    {layer:<26}: 无触发  -> 该层被完全吸收")
            else:
                top3 = sorted(items, key=lambda x: x[1], reverse=True)[:3]
                detail = ", ".join([f"{v}({r:.3f})" for v, r in top3])
                print(f"    {layer:<26}: {detail}")
    
    print("\n" + "=" * 80)

# -------------------------------------------------
# 7. 主控函数（一键运行）
# -------------------------------------------------
def main_experiment_2(test_cases=None,
                      save_path='exp2_perturbation_results.pkl',
                      test_suite_path=None):
    """
    实验二主控函数（修正版）。
    
    调用方式：
        main_experiment_2(tests)                                   # 内存直传
        main_experiment_2(tests, 'exp2_results.pkl')               # 同时指定保存路径
        main_experiment_2(test_suite_path='fixed_test_suite.pkl')  # 从文件加载
    """
    # 8.1 获取测试用例
    if test_cases is not None:
        print(f"[实验二] 使用内存传入的测试套件 (n={len(test_cases)})")
    elif test_suite_path is not None:
        try:
            with open(test_suite_path, 'rb') as f:
                suite_data = pickle.load(f)
            test_cases = suite_data['test_cases']
            print(f"[实验二] 已加载固定测试套件: {test_suite_path} (n={len(test_cases)})")
        except FileNotFoundError:
            print(f"[实验二] 错误：未找到 {test_suite_path}")
            return
    else:
        print("[实验二] 错误：必须提供 test_cases 或 test_suite_path 之一")
        return
    
    # 8.2 加载原始 Oracle
    oracle_func = load_oracle()
    if oracle_func is None:
        print("[实验二] Oracle 加载失败，请检查 M00.py 路径。")
        return
    
    # 8.3 配置扰动（含干净基线）
    perturbation_configs = {
        'clean':                {},
        'gaussian_noise':       {'sigma': 0.1},
        'extreme_boundary':     {'p': 0.05},
        'adversarial_shift':    {'delta': 0.01},
        'precision_degradation': {},
    }
    
    # 8.4 运行实验
    results = run_saturation_experiment(test_cases, oracle_func, perturbation_configs)
    
    # 8.5 统计检验
    clean_res = results['clean']
    stat_reports = {}
    for pert_name in ['gaussian_noise', 'extreme_boundary',
                      'adversarial_shift', 'precision_degradation']:
        stat_reports[pert_name] = paired_saturation_test(clean_res, results[pert_name])
    
    # 8.6 输出报表
    print_saturation_report(results, stat_reports)
    
    # 8.7 保存结果
    payload = {
        'results': results,
        'stat_reports': stat_reports,
        'layer_map': LAYER_MAP,
        'perturbation_configs': perturbation_configs,
        'metadata': 'Exp2: Input Perturbation Saturation Absorption'
    }
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n[实验二] 完整结果已保存至: {save_path}")
    return payload

#endregion


if __name__=='__main__':   

    # 定义四分类与违规分类映射
    categories = {
        'Numerical Stability': [1, 2, 5, 12, 16, 17, 18],      
        'Statistical Moments': [8, 9, 10, 11],  
        'Distributional Axiom': [3, 4, 6, 7, 13,14],
        'Structural Invariants': [0,15, 19]                   
    }

    # tests=generate_and_save_test_suite(n=200)  #用于首次生成测试用例，然后固定下来，测试N次试验。
    # # exit()
    # mutant_files = [f"M{i:02d}.py" for i in range(1, 64)]
    # tests = load_fixed_test_suite()
    
    # mutant_funcs = load_mutants(mutant_files)
    # kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutant_funcs, tests)
    f_p=load_fingerprints()
    kill_matrix=f_p["kill_matrix"]
    violation_map=f_p['violation_map']

#region RQ1 experiment CM Metrics
    # plot_cm_both(kill_matrix,violation_map)
    # plot_km_fp_confusion_heatmap(kill_matrix,violation_map)
    # matrix=plot_km_layer_heatmap(kill_matrix,violation_map,categories) #这个有问题
    # print(matrix)
#endregion
    
#region RQ1 significance test
    # a=analyze_single_operator('softmax',kill_matrix,violation_map)
    # print_operator_summary(a)
    plot_rq1_McNemar_test()
#endregion
    
#region RQ2:experiment A-C
    # print('RQ2:experiment A: Metrics')
    # v=compute_rq2_metrics(kill_matrix, violation_map,categories)
    # print(v)

    # print('RQ2:experiment B: fingerprint tsne')
    # plot_fingerprint_tsne(violation_map,categories,save_path='rq2\tsne.png')

    # print('RQ2:experiment C: 3 Cases ')
    # cases=extract_case_studies(kill_matrix,violation_map,categories)
    # for c in cases:
    #     print(f"\nCase: {c['m1']} vs {c['m2']}")
    #     print(f"  KM pattern: {c['km_pattern'][:5]}... (same class)")
    #     print(f"  FP({c['m1']}): {c['fp_m1']} → {c['layer_name_m1']} (L{c['dominant_m1']})")
    #     print(f"  FP({c['m2']}): {c['fp_m2']} → {c['layer_name_m2']} (L{c['dominant_m2']})")    
    # plot_cases()
#endregion
   
#region RQ3: experiment
    # print('RQ3: experiment')    
    # results = run_rq3_experiment(kill_matrix, violation_map, categories)
    # for strategy, metrics in results.items():
    #     print(f"\n{strategy}:")
    #     for k, v in metrics.items():
    #         print(f"  {k}: {v}")
    # plot_HVAR_by_qr3_data()
#endregion

#region RQ4 experiment A
# print('RQ4 experiment A: CI Interception Rate Validation')
# print('='*40)
# thresholds = [0.80, 0.85, 0.90, 0.95, 1.00]
# for th in thresholds:
#     result = run_rq4_experiment_a(
#         kill_matrix, violation_map, categories,
#         n_fine=20, survival_rate_threshold=th, debug=True
#     )

# print(f"Stage 1 Passed: {result['operator_summary']['stage1_passed']}")
# print(f"Stage 1 Failed: {result['operator_summary']['stage1_failed']}")
# print(f"Stage 2 Intercepted: {result['operator_summary']['stage2_intercepted']}")
# print(f"Stage 2 Clean Pass: {result['operator_summary']['stage2_clean_pass']}")
# print(f"IR: {result['core_metrics']['Interception_Rate_IR']:.2%}")
# print(f"CPR: {result['core_metrics']['Clean_Pass_Rate_CPR']:.2%}")
# plot_coverage(kill_matrix, violation_map)
# print('='*40)
# sr_vals = []
# for n in kill_matrix:
#     km = np.asarray(kill_matrix[n])
#     sr = np.mean(km == 0)
#     sr_vals.append((n, sr))

# # 按存活率降序排列
# sr_sorted = sorted(sr_vals, key=lambda x: x[1], reverse=True)

# print("=== 存活率 Top 35 ===")
# for i, (n, sr) in enumerate(sr_sorted[:35]):
#     flag = " >=0.95" if sr >= 0.95 else " >=0.90" if sr >= 0.90 else ""
#     print(f"{n}: {sr:.4f}{flag}")

# print(f"\n=== 关键统计 ===")
# print(f"存活率 >= 0.95: {sum(1 for _, sr in sr_vals if sr >= 0.95)}")
# print(f"存活率 >= 0.90: {sum(1 for _, sr in sr_vals if sr >= 0.90)}")
# print(f"存活率 >= 1.00: {sum(1 for _, sr in sr_vals if sr >= 1.00)}")
# print(f"最高存活率: {max(sr for _, sr in sr_vals):.4f}")
#endregion

#region RQ4 experiment B
# 1. 运行实验 A（固定 threshold，非循环）
# result_a = run_rq4_experiment_a(
#     kill_matrix, violation_map, categories,
#     n_fine=20, 
#     survival_rate_threshold=0.90,   # Softmax / LayerNorm 用 0.90
#     debug=False                       # 关闭调试输出
# )


# # 2. 提取 intercepted 变异体列表
# intercepted_mutants = result_a['intercepted_analysis']['intercepted_ids']

# print(f"实验 A 拦截变异体数: {len(intercepted_mutants)}")
# print(f"示例 ID: {intercepted_mutants[:5]}")

# # 3. 直接传入实验 B
# result_b = run_rq4_experiment_b(
#     intercepted_mutants=intercepted_mutants,
#     violation_map=violation_map,
#     categories=categories,
#     n_fine=20
# )

# # 4. 打印实验 B 核心指标
# print(f"样本量: {result_b['sample_size']}")
# print(f"DSC_KM={result_b['granularity']['DSC_KM']}")
# print(f"DSC_FP_strict={result_b['granularity']['DSC_FP_strict']}")
# print(f"DSC_FP_binned={result_b['granularity']['DSC_FP_binned']}")
# print(f"MLCR={result_b['granularity']['MLCR']}")
# print(f"DE_FP={result_b['diagnostic_entropy']['DE_FP_raw']:.3f} bits")
# print(f"DE_normalized={result_b['diagnostic_entropy']['DE_FP_normalized']:.3f}")
# print(f"Entropy gain={result_b['diagnostic_entropy']['entropy_gain']:.3f} bits")

# for case in result_b['case_reports']:
#     print(f"\n--- 案例: {case['mutant_1']} vs {case['mutant_2']} ---")
#     print(f"主导层: {case['dominant_layer']}")
#     print(f"Kill-Matrix: {case['km_diagnosis']}")
#     print(f"指纹差异: L1距离={case['l1_distance']}")
#     print(f"  {case['mutant_1']}: {case['fp_insight_m1']}")
#     print(f"  {case['mutant_2']}: {case['fp_insight_m2']}")
#endregion

#region QR_4_PLOT
# experiment_qr4_plot()
#endregion