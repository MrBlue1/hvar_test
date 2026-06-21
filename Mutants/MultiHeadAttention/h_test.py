import math
import itertools
import numpy as np
import random
from typing import List, Union, Optional, Any, Tuple, Dict, Callable
import importlib.util
import sys
import io
import os
import time
import warnings
from pathlib import Path
from contextlib import contextmanager
import copy
from Mutants.mutant_ananlysis import detect_fingerprint_collisions, detect_fingerprint_collisions2,print_collision_report,print_collision_report2, run_optimized_analysis,run_violation_statistc,collid_graph_t,run_vdr_budget_curve,print_operator_summary,analyze_single_operator

from Mutants.experiment_1 import plot_coverage,plot_cm_both,plot_km_fp_confusion_heatmap,plot_km_layer_heatmap

from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies
from Mutants.experiment_rq3 import run_rq3_experiment,run_rq3_experiment_debug
from Mutants.experiment_rq4 import run_rq4_experiment_a,run_rq4_experiment_b


# ==================== 1. 违规类型定义（18个，全部可触发） ====================
ALL_BEHAVIOR_TYPES = [
    "nan_output",                      # 0
    "inf_output",                      # 1
    "overflow_warning",                # 2
    "underflow_warning",               # 3
    "division_by_zero",                # 4
    "sqrt_negative",                   # 5
    "attention_weights_sum_not_one",   # 6
    "attention_weights_negative",      # 7
    "attention_weights_all_same",      # 8
    "variance_too_high",               # 9
    "attention_weights_all_zero",      # 10
    "shape_mismatch_error",            # 11
    "d_model_not_divisible",           # 12
    "seq_len_mismatch",                # 13
    "broadcast_failure",               # 14
    "d_k_zero",                        # 15
    "weight_matrix_not_initialized",   # 16
    "mask_all_invalid",                # 17
    "log_zero_warning"                 # 18
]


def rand_tensor(shape, low=-1.0, high=1.0):
    """生成均匀分布的随机张量"""
    return np.random.uniform(low, high, shape)

def normal_tensor(shape, mean=0.0, std=1.0):
    """生成正态分布的随机张量"""
    return np.random.randn(*shape) * std + mean

def generate_test_cases(n: int, seed: Optional[int] = None,
                        base_d_model=8, base_num_heads=2,
                        base_batch_size=3, base_seq_len=4):
    if seed is not None:
        np.random.seed(seed)
    n1 = int(n * 0.4)
    n2 = int(n * 0.4)
    n3 = n - n1 - n2
    cases = []
    # 策略1: 常规
    for _ in range(n1):
        query = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        key = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        value = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        mask = None if np.random.rand() > 0.3 else (np.random.rand(base_batch_size, base_seq_len, base_seq_len) > 0.2).astype(np.float32)
        cases.append({
            "model_params": {"d_model": base_d_model, "num_heads": base_num_heads},
            "inputs": {"query": query, "key": key, "value": value, "mask": mask},
            "metadata": {"strategy": "normal"}
        })
    # 策略2: 针对性触发每种违规类型（至少2个）
    # 使用字典记录每种类型对应的生成函数
    def add_case(vtype, make_inputs, make_params=None):
        for _ in range(2):
            params = make_params() if make_params else {"d_model": base_d_model, "num_heads": base_num_heads}
            cases.append({
                "model_params": params,
                "inputs": make_inputs(),
                "metadata": {"strategy": "targeted", "target_type": vtype}
            })
    # nan_output
    add_case("nan_output", lambda: {
        "query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
        "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
        "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
        "mask": None
    })
    # 手动注入nan
    def inject_nan():
        q = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        q[0,0,0] = np.nan
        return {"query": q, "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "value": np.random.randn(base_batch_size, base_seq_len, base_d_model), "mask": None}
    add_case("nan_output", inject_nan)
    # inf_output
    def inject_inf():
        q = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        q[0,0,0] = np.inf
        return {"query": q, "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "value": np.random.randn(base_batch_size, base_seq_len, base_d_model), "mask": None}
    add_case("inf_output", inject_inf)
    # overflow_warning: 极大值
    def huge():
        scale = 1e30
        return {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model) * scale,
                "key": np.random.randn(base_batch_size, base_seq_len, base_d_model) * scale,
                "value": np.random.randn(base_batch_size, base_seq_len, base_d_model) * scale,
                "mask": None}
    add_case("overflow_warning", huge)
    # underflow_warning: 极小值
    def tiny():
        scale = 1e-300
        return {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model) * scale,
                "key": np.random.randn(base_batch_size, base_seq_len, base_d_model) * scale,
                "value": np.random.randn(base_batch_size, base_seq_len, base_d_model) * scale,
                "mask": None}
    add_case("underflow_warning", tiny)
    # division_by_zero: 需要变异体，测试用例用正常输入
    add_case("division_by_zero", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                          "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                          "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                          "mask": None})
    # sqrt_negative: 需要变异体
    add_case("sqrt_negative", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                       "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                       "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                       "mask": None})
    # attention_weights_sum_not_one: 需变异体
    add_case("attention_weights_sum_not_one", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                       "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                       "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                       "mask": None})
    # attention_weights_negative: 需变异体
    add_case("attention_weights_negative", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                    "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                    "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                    "mask": None})
    # attention_weights_all_same: 常数输入
    def const_input():
        c = 1.0
        return {"query": np.full((base_batch_size, base_seq_len, base_d_model), c),
                "key": np.full((base_batch_size, base_seq_len, base_d_model), c),
                "value": np.full((base_batch_size, base_seq_len, base_d_model), c),
                "mask": None}
    add_case("attention_weights_all_same", const_input)
    # variance_too_high: 极大值输入
    add_case("variance_too_high", huge)
    # attention_weights_all_zero: 需要变异体
    add_case("attention_weights_all_zero", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                    "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                    "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                    "mask": None})
    # shape_mismatch_error: 形状不匹配
    def shape_mismatch():
        return {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "key": np.random.randn(base_batch_size, base_seq_len+1, base_d_model),
                "value": np.random.randn(base_batch_size, base_seq_len+1, base_d_model),
                "mask": None}
    add_case("shape_mismatch_error", shape_mismatch)
    # d_model_not_divisible: 参数不整除
    def not_divisible():
        return {"d_model": base_d_model+1, "num_heads": base_num_heads}
    add_case("d_model_not_divisible", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model+1),
                                               "key": np.random.randn(base_batch_size, base_seq_len, base_d_model+1),
                                               "value": np.random.randn(base_batch_size, base_seq_len, base_d_model+1),
                                               "mask": None},
             make_params=lambda: {"d_model": base_d_model+1, "num_heads": base_num_heads})
    # seq_len_mismatch: QK序列长度不同
    def seq_mismatch():
        return {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "key": np.random.randn(base_batch_size, base_seq_len+2, base_d_model),
                "value": np.random.randn(base_batch_size, base_seq_len+2, base_d_model),
                "mask": None}
    add_case("seq_len_mismatch", seq_mismatch)
    # broadcast_failure: mask形状错误
    def bad_mask():
        mask = np.random.rand(base_batch_size, base_seq_len, base_seq_len+1)
        return {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "mask": mask}
    add_case("broadcast_failure", bad_mask)
    # d_k_zero: 模型参数导致d_k=0
    def dk_zero_params():
        dmod = 4
        nheads = 8
        return {"d_model": dmod, "num_heads": nheads}
    add_case("d_k_zero", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, 4),
                                  "key": np.random.randn(base_batch_size, base_seq_len, 4),
                                  "value": np.random.randn(base_batch_size, base_seq_len, 4),
                                  "mask": None},
             make_params=dk_zero_params)
    # weight_matrix_not_initialized: 需变异体SDL
    add_case("weight_matrix_not_initialized", lambda: {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                       "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                       "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                                                       "mask": None})
    # mask_all_invalid: mask全0
    def allzero_mask():
        mask = np.zeros((base_batch_size, base_seq_len, base_seq_len))
        return {"query": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "key": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "value": np.random.randn(base_batch_size, base_seq_len, base_d_model),
                "mask": mask}
    add_case("mask_all_invalid", allzero_mask)
    # log_zero_warning: 全负无穷输入
    def all_neg_inf():
        q = np.full((base_batch_size, base_seq_len, base_d_model), -np.inf)
        return {"query": q, "key": q, "value": q, "mask": None}
    add_case("log_zero_warning", all_neg_inf)
    # 策略2数量可能超过n2，后面会随机选择到n2个
    # 策略3: 边界攻击
    attack_modes = ["extreme_large", "extreme_small", "nan_injection", "inf_injection", "zero_masked", "negative_values"]
    for _ in range(n3):
        mode = np.random.choice(attack_modes)
        query = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        key = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        value = np.random.randn(base_batch_size, base_seq_len, base_d_model)
        mask = None
        if mode == "extreme_large":
            query *= 1e30
            key *= 1e30
        elif mode == "extreme_small":
            query *= 1e-300
            key *= 1e-300
        elif mode == "nan_injection":
            query[0,0,0] = np.nan
        elif mode == "inf_injection":
            query[0,0,0] = np.inf
        elif mode == "zero_masked":
            mask = np.zeros((base_batch_size, base_seq_len, base_seq_len))
        elif mode == "negative_values":
            query = -np.abs(query)
        cases.append({
            "model_params": {"d_model": base_d_model, "num_heads": base_num_heads},
            "inputs": {"query": query, "key": key, "value": value, "mask": mask},
            "metadata": {"strategy": "boundary", "mode": mode}
        })
    # 裁剪到n个
    if len(cases) > n:
        cases = np.random.choice(cases, n, replace=False).tolist()
    while len(cases) < n:
        cases.append(cases[-1].copy())
    return cases



# =========================================================
# 载入 Oracle & Mutants
# =========================================================
def load_oracle():
    spec = importlib.util.spec_from_file_location("M00", "mutants/M00.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiHeadAttention

def load_mutants():
    mutants = {}
    for fname in os.listdir("mutants"):
        if fname.startswith("M") and fname.endswith(".py") and fname != "M00.py":
            mid = fname[:-3]
            try:
                spec = importlib.util.spec_from_file_location(mid, os.path.join("mutants", fname))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                mutants[mid] = module.MultiHeadAttention
            except Exception as e:
                print(f"Warning: failed to load {fname}: {e}")
    return mutants

# 数值容差默认值
DEFAULT_TOL = 1e-6

# ---------- 1. detect_violations ----------
def detect_violations(test_case, S, tol=1e-6):
    violations = []
    if S is None:
        return violations
    output, attn_weights = S
    inputs = test_case["inputs"]
    # NaN / Inf
    if np.isnan(output).any() or np.isnan(attn_weights).any():
        violations.append("nan_output")
    if np.isinf(output).any() or np.isinf(attn_weights).any():
        violations.append("inf_output")
    # underflow / overflow 通过数值范围判断
    if np.any((np.abs(output) < 1e-300) & (output != 0)) or np.any((np.abs(attn_weights) < 1e-300) & (attn_weights != 0)):
        violations.append("underflow_warning")
    if np.any(np.abs(output) > 1e30) or np.any(np.abs(attn_weights) > 1e30):
        violations.append("overflow_warning")
    # attention_weights_sum_not_one
    if not np.allclose(np.sum(attn_weights, axis=-1), 1.0, rtol=tol, atol=tol):
        violations.append("attention_weights_sum_not_one")
    # attention_weights_negative
    if np.any(attn_weights < -tol):
        violations.append("attention_weights_negative")
    # attention_weights_all_same
    # 只检查第一个batch和head的第一个query位置
    if attn_weights.size > 0:
        sample = attn_weights[0,0,0,:]
        if np.allclose(sample, sample[0], atol=tol):
            violations.append("attention_weights_all_same")
    # variance_too_high
    if np.var(output) > 10.0:
        violations.append("variance_too_high")
    # attention_weights_all_zero
    if np.allclose(attn_weights, 0.0, atol=tol):
        violations.append("attention_weights_all_zero")
    # shape_mismatch_error 和 seq_len_mismatch: 这里无法检测，由异常处理
    # d_model_not_divisible: 检测模型参数
    params = test_case["model_params"]
    if params["d_model"] % params["num_heads"] != 0:
        violations.append("d_model_not_divisible")
    # seq_len_mismatch: 检查输入形状
    if inputs["query"].shape[1] != inputs["key"].shape[1]:
        violations.append("seq_len_mismatch")
    # broadcast_failure: mask形状错误
    mask = inputs.get("mask")
    if mask is not None:
        expected = (inputs["query"].shape[0], inputs["query"].shape[1], inputs["key"].shape[1])
        if mask.shape != expected:
            violations.append("broadcast_failure")
    # d_k_zero: 检查模型参数
    d_k = params["d_model"] // params["num_heads"]
    if d_k == 0:
        violations.append("d_k_zero")
    # mask_all_invalid
    if mask is not None and np.all(mask == 0):
        violations.append("mask_all_invalid")
    # log_zero_warning: 检查输入是否包含 -inf
    if np.any(np.isneginf(inputs["query"])) or np.any(np.isneginf(inputs["key"])):
        violations.append("log_zero_warning")
    # 增强 inf_output 检测：输入包含 inf 且输出出现 nan/inf 时标记
    inputs = test_case["inputs"]
    if (np.any(np.isinf(inputs["query"])) or 
        np.any(np.isinf(inputs["key"])) or 
        np.any(np.isinf(inputs["value"]))):
        if "nan_output" in violations or "inf_output" in violations:
            violations.append("inf_output")

    # 以下由变异体触发，这里不检测
    # division_by_zero, sqrt_negative, weight_matrix_not_initialized 会在 run_test 中通过异常/警告捕获
    return list(set(violations))


# ---------- 2. layered_decision_engine ----------
def layered_decision_engine(test_case: Dict, S_O: Any, S_M: Any, tol: float = DEFAULT_TOL) -> Tuple[bool, List[str], List[str]]:
    """
    决策引擎，判断变异体是否被杀死。
    参数：
        test_case: 测试用例
        S_O: 原始模型输出
        S_M: 变异体输出
        tol: 数值容差
    返回：
        killed, oracle_viol, mutant_viol
    """
    oracle_viol = detect_violations(test_case, S_O, tol)
    mutant_viol = detect_violations(test_case, S_M, tol)
    
    # 如果都没有违规，则比较输出值是否相等
    if len(oracle_viol) == 0 and len(mutant_viol) == 0:
        # 比较输出值（需要解包并比较 output 和 attention_weights）
        if isinstance(S_O, tuple) and isinstance(S_M, tuple) and len(S_O) == len(S_M):
            all_equal = True
            for a, b in zip(S_O, S_M):
                if not np.allclose(a, b, rtol=tol, atol=tol):
                    all_equal = False
                    break
            killed = not all_equal
        else:
            # 输出结构不一致视为不同
            killed = True
    else:
        # 至少有一方违规，则只要违规情况不完全相同就杀死
        # 如果双方违规类型集合相同，则不算杀死（因为行为一致）
        if set(oracle_viol) == set(mutant_viol):
            killed = False
        else:
            killed = True
    
    not np.allclose(a, b, rtol=tol, atol=tol)
    # return killed, oracle_viol, mutant_viol


# ---------- 3. run_test ----------
def run_test(oracle_cls, mutant_cls, test_case):
    def run_model(cls):
        captured_warnings = []
        def warn_catcher(message, category, filename, lineno, file=None, line=None):
            msg = str(message)
            if "invalid value" in msg and "sqrt" in msg:
                captured_warnings.append("sqrt_negative")
            elif "invalid value" in msg:
                captured_warnings.append("nan_output")
            elif "divide by zero" in msg:
                captured_warnings.append("division_by_zero")
            elif "overflow" in msg:
                captured_warnings.append("overflow_warning")
            elif "underflow" in msg:
                captured_warnings.append("underflow_warning")
            else:
                captured_warnings.append("invalid_output")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.showwarning = warn_catcher
            try:
                model = cls(**test_case["model_params"])
                S = model.forward(**test_case["inputs"])
                # 如果输出中有 nan/inf 且未捕获，手动添加
                if S is not None:
                    out, attn = S
                    if np.isnan(out).any() or np.isnan(attn).any():
                        if "nan_output" not in captured_warnings:
                            captured_warnings.append("nan_output")
                    if np.isinf(out).any() or np.isinf(attn).any():
                        if "inf_output" not in captured_warnings:
                            captured_warnings.append("inf_output")
                return S, list(set(captured_warnings))
            except AssertionError as e:
                if "d_model 必须能被 num_heads 整除" in str(e):
                    return None, ["d_model_not_divisible", "d_k_zero"]
                return None, ["invalid_output"]
            except ValueError as e:
                if "shape" in str(e).lower():
                    return None, ["shape_mismatch_error", "seq_len_mismatch"]
                return None, ["invalid_output"]
            except NameError as e:
                if "W_q" in str(e) or "not defined" in str(e):
                    return None, ["weight_matrix_not_initialized"]
                return None, ["invalid_output"]
            except Exception:
                return None, ["invalid_output"]
    S_oracle, viol_oracle = run_model(oracle_cls)
    S_mutant, viol_mutant = run_model(mutant_cls)
    # 检测函数补充数学违规
    if S_oracle is not None:
        viol_oracle += detect_violations(test_case, S_oracle)
    if S_mutant is not None:
        viol_mutant += detect_violations(test_case, S_mutant)
    viol_oracle = list(set(viol_oracle))
    viol_mutant = list(set(viol_mutant))
    # 判断是否杀死
    if set(viol_oracle) == set(viol_mutant):
        killed = False
    else:
        killed = True
    return killed, viol_oracle, viol_mutant



# ---------- 4. build_fingerprints ----------
def build_fingerprints(mutants, test_cases, oracle_cls):
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}
    for mid, mcls in mutants.items():
        killed_list = []
        viol_count = np.zeros(len(ALL_BEHAVIOR_TYPES))
        for tc in test_cases:
            killed, _, viol_m = run_test(oracle_cls, mcls, tc)
            killed_list.append(1 if killed else 0)
            for v in viol_m:
                if v in ALL_BEHAVIOR_TYPES:
                    viol_count[ALL_BEHAVIOR_TYPES.index(v)] += 1
        fingerprints[mid] = np.array(killed_list)
        ms_per_mutant[mid] = np.mean(killed_list)
        violation_map[mid] = viol_count
    return fingerprints, ms_per_mutant, violation_map


def check_violation_coverage(violation_map):
    hit = [False]*len(ALL_BEHAVIOR_TYPES)
    for counts in violation_map.values():
        for i, c in enumerate(counts):
            if c > 0:
                hit[i] = True
    hit_types = [ALL_BEHAVIOR_TYPES[i] for i, h in enumerate(hit) if h]
    missing = [ALL_BEHAVIOR_TYPES[i] for i, h in enumerate(hit) if not h]
    coverage = 100 * len(hit_types) / len(ALL_BEHAVIOR_TYPES)
    return hit_types, missing, coverage



if __name__ == "__main__":    
    # 生成测试用例
    print("\n1. 生成测试用例...")
    test_cases = generate_test_cases(200, seed=42)  # 增加测试用例数量
    print(f"   生成了 {len(test_cases)} 个测试用例")

    # 加载原始代码
    print("\n2. 加载原始代码...")
    oracle_class = load_oracle()   # 返回类，不是实例
    if oracle_class is None:
        print("   错误：无法加载原始代码 M00.py")
        sys.exit(1)
    print("   ✓ 原始代码加载成功")
    
    # 加载所有变异体    
    print("\n3. 加载变异体...")
    mutants = load_mutants()        # 返回 dict {mutant_id: class}
    print(f"   ✓ 已加载 {len(mutants)} 个变异体")
    
    # 构建指纹
    print("\n4. 执行测试并构建指纹...")
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(
        mutants,
        test_cases,
        oracle_class
    )
    
    categories = {
        'Numerical Stability': [1, 2, 5, 12, 16, 17, 18],
        'Statistical Moments': [8, 9, 10, 11],
        'Distributional Axiom': [3, 4, 6, 7, 13, 14],
        'Structural Invariants': [0, 15]
    }
    

    # a=analyze_single_operator('MultiHeadAttention',kill_matrix,violation_map)
    # print_operator_summary(a)


    #region RQ2:experiment A-C
    # print('RQ2:experiment A: Metrics')
    # v=compute_rq2_metrics(kill_matrix, violation_map,categories)
    # print(v)

    # # print('RQ2:experiment B: fingerprint tsne')
    # # plot_fingerprint_tsne(violation_map,categories,save_path='rq2\tsne.png')

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
    print('RQ3: experiment')    
    results = run_rq3_experiment_debug(kill_matrix, violation_map, categories, target_ratio=0.25, random_seed=42)
    print(result['Global_KMeans'])
    # for strategy, metrics in results.items():
    #     print(f"\n{strategy}:")
    #     for k, v in metrics.items():
    #         print(f"  {k}: {v}")
    
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

