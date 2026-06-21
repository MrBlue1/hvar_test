import numpy as np
import random
from typing import List, Union, Optional, Any, Tuple, Dict, Callable
import importlib.util
import os
import warnings
import copy
from Mutants.mutant_ananlysis import detect_fingerprint_collisions, detect_fingerprint_collisions2,print_collision_report,print_collision_report2, run_optimized_analysis,run_violation_statistc,collid_graph_t,run_vdr_budget_curve,print_operator_summary,analyze_single_operator

from Mutants.experiment_1 import plot_coverage,plot_cm_both,plot_km_fp_confusion_heatmap,plot_km_layer_heatmap

from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies,plot_cases
from Mutants.experiment_rq3 import run_rq3_experiment
from Mutants.experiment_rq4 import run_rq4_experiment_a,run_rq4_experiment_b
all_behavior_types = [
    # ========= 基础运行异常 =========
    "invalid_output",        # 输出为None等非法
    "type_error",            # 输出无法转为数值（如字符串、嵌套结构）
    "exception",             # 运行抛异常

    # ========= 数值异常 =========
    "nan",                   # 出现NaN
    "inf",                   # 出现正/负无穷
    "overflow",              # 数值过大（>1e308）
    "underflow",             # 非零但接近0（数值下溢）

    # ========= ReLU核心语义违规 =========
    "negative_output",       # 输出出现负数（绝对错误）
    "positive_clipped",      # 输入>0但输出不等于输入
    "negative_not_zero",     # 输入<=0但输出不为0

    # ========= 输入输出结构一致性 =========
    "shape_mismatch",        # 输入list但输出长度变化
    "type_mismatch",         # 输入list却输出标量，或反之

    # ========= 边界/阈值问题 =========
    "threshold_error",       # >0 / >=0 判定错误（通过行为间接体现）
    "zero_unexpected",       # 异常出现0（弱信号，用于辅助判断）

    # ========= Warning映射 =========
    "overflow_warning",      # numpy overflow warning
    "underflow_warning",     # numpy underflow warning
    "invalid_warning",       # invalid value warning（如0/0）

    # ========= 兜底 =========
    "unexpected_behavior"    # 未分类异常（极少触发）
]

np.seterr(all='warn')
warnings.filterwarnings("ignore")

# ===== ReLU 原始函数（用于保证策略1合法）=====
def relu(x):
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0


# ===== 主函数 =====
def generate_lhs_samples(n: int,
                         seed: Optional[int] = None,
                         max_dim: int = 10) -> List[Union[float, list]]:
    if seed is not None:
        np.random.seed(seed)

    samples = []

    n1 = int(n * 0.4)
    n2 = int(n * 0.4)
    n3 = n - n1 - n2

    # =========================================================
    # 策略1：常规测试（40%）
    # =========================================================
    normal_samples = []
    for _ in range(n1):
        if np.random.rand() < 0.5:
            # 标量
            if np.random.rand() < 0.1:
                # 极端值
                x = np.random.choice([1e-12, -1e-12, 1e6, -1e6, 0])
            else:
                x = np.random.uniform(-10, 10)
        else:
            # list
            size = np.random.randint(1, max_dim)
            if np.random.rand() < 0.1:
                vals = np.random.choice([1e-12, -1e-12, 1e6, -1e6, 0], size=size)
            else:
                vals = np.random.uniform(-10, 10, size=size)
            x = vals.tolist()

        # 保证原始代码可运行
        try:
            relu(x)
            normal_samples.append(x)
        except:
            continue

    # =========================================================
    # 策略2：故障触发（40%）
    # =========================================================
    targeted_samples = []

    # 覆盖违规类型（每种2个）
    behavior_generators = [
        lambda: -np.random.uniform(1, 10),                 # negative_output
        lambda: [np.random.uniform(-10, -1) for _ in range(5)],

        lambda: 0,                                         # zero_misclassified
        lambda: [0 for _ in range(5)],

        lambda: 1e308,                                     # overflow
        lambda: -1e308,

        lambda: 1e-308,                                    # underflow
        lambda: -1e-308,

        lambda: [1e308, -1e308, 0],                        # mixed extreme
        lambda: [1e-308, -1e-308, 0],

        lambda: [np.nan, 1, -1],                           # nan
        lambda: [np.inf, -np.inf],                         # inf

        lambda: [],                                        # shape edge
        lambda: [1],                                       # minimal case

        lambda: [[1, -1], [2, -2]],                         # unsupported_input
        lambda: "string_input",                            # type_error
    ]

    for gen in behavior_generators:
        try:
            targeted_samples.append(gen())
        except:
            continue

    # ===== 补充强制触发 =====
    targeted_samples.extend([
        1e10,              # overflow
        -1e10,
        1e-15,             # underflow
        -1e-15,
        [1e10, -1e10],
        [1e-15, -1e-15],
        [np.nan],
        [np.inf],
        [[1,2],[3,4]],     # unexpected_behavior
    ])
    # 扩充到 40%
    while len(targeted_samples) < n2:
        base = targeted_samples[np.random.randint(len(targeted_samples))]

        try:
            if isinstance(base, list):
                arr = np.array(base, dtype=float, copy=True)
                noise = np.random.randn(*arr.shape) * 0.01
                new_sample = (arr + noise).tolist()
            elif isinstance(base, (int, float)):
                new_sample = base + np.random.randn() * 0.01
            else:
                continue

            targeted_samples.append(new_sample)
        except:
            continue

    # =========================================================
    # 策略3：边界攻击（20%）
    # =========================================================
    attack_samples = []

    for _ in range(n3):
        mode = np.random.choice([
            'overflow',
            'underflow',
            'precision',
            'mixed_scale'
        ])

        if mode == 'overflow':
            x = np.random.uniform(1e10, 1e308)

        elif mode == 'underflow':
            x = np.random.uniform(-1e-308, 1e-308)

        elif mode == 'precision':
            x = np.random.choice([1e-16, -1e-16, 1e-15, -1e-15])

        elif mode == 'mixed_scale':
            size = np.random.randint(2, max_dim)
            vals = []
            for _ in range(size):
                vals.append(np.random.choice([
                    np.random.uniform(-1e-10, 1e-10),
                    np.random.uniform(-1e10, 1e10)
                ]))
            x = vals

        attack_samples.append(x)

    # =========================================================
    # 合并 & 裁剪
    # =========================================================
    all_samples = normal_samples + targeted_samples + attack_samples

    if len(all_samples) > n:
        idx = np.random.choice(len(all_samples), n, replace=False)
        all_samples = [all_samples[i] for i in idx]

    return all_samples

# =========================================================
# 载入 Oracle & Mutants
# =========================================================

def load_oracle():
    """
    加载原始代码 M00.py 中的 relu 函数
    """
    try:
        path = os.path.join("mutants", "M00.py")
        spec = importlib.util.spec_from_file_location("M00", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.relu
    except Exception as e:
        print(f"[WARN] Failed to load Oracle (M00): {e}")
        return None


def load_mutants(start=1, end=65):
    """
    加载 M01.py ~ M65.py 的变异体
    返回: dict { 'M01': relu_func, ... }
    """
    mutant_funcs = {}

    for i in range(start, end + 1):
        name = f"M{i:02d}"
        path = os.path.join("mutants", f"{name}.py")

        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue

        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # ⚠️ 强制检查是否有 relu 函数
            if not hasattr(mod, "relu"):
                print(f"[WARN] {name} has no 'relu' function")
                continue

            mutant_funcs[name] = mod.relu

        except Exception as e:
            print(f"[WARN] Failed to load {name}: {e}")

    return mutant_funcs


# ==========================
# 三层决策引擎
# ==========================

def detect_violations(X, S, tol=1e-8):

    viols = []
    # ========= 空输出处理 =========
    if isinstance(S, list) and len(S) == 0:
        return ["invalid_output"]
    
    # ========= 输出非法 =========
    if S is None:
        return ["invalid_output"]

    # ========= 类型 =========
    if isinstance(S, list):
        try:
            arr = np.array(S, dtype=float)
        except:
            return ["type_error"]
    elif isinstance(S, (int, float, np.number)):
        arr = np.array([S], dtype=float)
    else:
        return ["type_error"]
    
    # ========= 再次保护 =========
    if arr.size == 0:
        return ["invalid_output"]

    # ========= 数值异常 =========
    if np.any(np.isnan(arr)):
        viols.append("nan")
        viols.append("invalid_warning")  # 补充触发

    if np.any(np.isinf(arr)):
        viols.append("inf")
        viols.append("overflow_warning")

    # ========= overflow / underflow（放宽条件）=========
    if np.any(np.abs(arr) > 1e6):   # ❗ 原来1e308太高
        viols.append("overflow")

    if np.any((np.abs(arr) < 1e-12) & (arr != 0)):
        viols.append("underflow")
        viols.append("underflow_warning")

    # ========= ReLU语义 =========
    if np.any(arr < -tol):
        viols.append("negative_output")

    # ========= zero =========
    if np.any(np.isclose(arr, 0, atol=tol)):
        viols.append("zero_unexpected")

    # ========= 输入输出关系 =========
    if isinstance(X, list) and isinstance(S, list):
        if len(X) != len(S):
            viols.append("shape_mismatch")

    if isinstance(X, list) != isinstance(S, list):
        viols.append("type_mismatch")

    # ========= 强化语义检测 =========
    if isinstance(X, list) and isinstance(S, list):
        #  只处理“平坦结构”
        if any(isinstance(i, list) for i in X):
            viols.append("unexpected_behavior")
            # return viols  # 或者标记 unexpected_behavior
        else:
            for x, s in zip(X, S):
                if x > 0 and not np.isclose(s, x, atol=tol, rtol=tol):
                    viols.append("positive_clipped")
                    viols.append("threshold_error")   #  触发点
                    break
                if x <= 0 and not np.isclose(s, 0, atol=tol):
                    viols.append("negative_not_zero")
                    break

    elif isinstance(X, (int, float)):
        s = arr[0]
        if X > 0 and not np.isclose(s, X, atol=tol, rtol=tol):
            viols.append("positive_clipped")
            viols.append("threshold_error")
        if X <= 0 and not np.isclose(s, 0, atol=tol):
            viols.append("negative_not_zero")

    # ========= 兜底 =========
    if not viols:
        # 检测“奇怪但合法”的情况
        if isinstance(S, list) and any(isinstance(i, list) for i in S):
            viols.append("unexpected_behavior")

    return list(set(viols))

def layered_decision_engine(X, S_O, S_M, tol=1e-5):
    oracle_viol = detect_violations(X, S_O, tol)
    mutant_viol = detect_violations(X, S_M, tol)

    
    if True:
        try:
            equal = np.allclose(
                np.array(S_O, dtype=float),
                np.array(S_M, dtype=float),
                atol=tol,
                equal_nan=True
            )
        except:
            equal = False

        return (not equal), oracle_viol, mutant_viol

    if not oracle_viol and not mutant_viol:
        try:
            equal = np.allclose(
                np.array(S_O, dtype=float),
                np.array(S_M, dtype=float),
                atol=tol,
                equal_nan=True
            )
        except:
            equal = False

        return (not equal), oracle_viol, mutant_viol


    if bool(oracle_viol) != bool(mutant_viol):
        return True, oracle_viol, mutant_viol

    if set(oracle_viol) != set(mutant_viol):
        return True, oracle_viol, mutant_viol

    return False, oracle_viol, mutant_viol


def map_warning(msg: str):
    """
    将 Python / numpy warning 映射到 all_behavior_types
    """
    msg = msg.lower()

    if "overflow" in msg:
        return "overflow_warning"
    elif "underflow" in msg:
        return "underflow_warning"
    elif "invalid" in msg or "divide" in msg:
        return "invalid_warning"
    else:
        return None


def run_test(oracle_func, mutant_func, test_case, tol=1e-8):
    """
    执行单个测试用例
    返回:
        killed: bool
        oracle_viol: list[str]
        mutant_viol: list[str]
    """

    S_O, S_M = None, None
    oracle_viol, mutant_viol = [], []

    # =========================================================
    # 1️⃣ 执行 Oracle
    # =========================================================
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            S_O = oracle_func(test_case)
            # 处理 warning
            for warn in w:
                mapped = map_warning(str(warn.message))
                if mapped:
                    oracle_viol.append(mapped)

    # except Exception:
    #     oracle_viol.append("exception")
    except Exception as e:
        if isinstance(e, TypeError):
            oracle_viol.append("type_error")
        else:
            oracle_viol.append("exception")

    # =========================================================
    # 2️⃣ 执行 Mutant
    # =========================================================
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            S_M = mutant_func(test_case)
            # 处理 warning
            for warn in w:
                mapped = map_warning(str(warn.message))
                if mapped:
                    mutant_viol.append(mapped)

    # except Exception:
    #     mutant_viol.append("exception")
    except Exception as e:
        if isinstance(e, TypeError):
            mutant_viol.append("type_error")
        else:
            mutant_viol.append("exception")

    # =========================================================
    # 3️⃣ 第一层判定（异常/警告层）
    # =========================================================

    # 情况1：双方都有违规
    if oracle_viol and mutant_viol:
        killed = set(oracle_viol) != set(mutant_viol)
        return killed, list(set(oracle_viol)), list(set(mutant_viol))

    # 情况2：只有一方违规
    if bool(oracle_viol) != bool(mutant_viol):
        return True, list(set(oracle_viol)), list(set(mutant_viol))

    # =========================================================
    # 4️⃣ 第二层判定（数值语义层）
    # =========================================================
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        killed, oracle_v2, mutant_v2 = layered_decision_engine(
            test_case, S_O, S_M, tol
        )

        for warn in w:
            mapped = map_warning(str(warn.message))
            if mapped:
                oracle_viol.append(mapped)
                mutant_viol.append(mapped)

    return (
        killed,
        list(set(oracle_viol + oracle_v2)),
        list(set(mutant_viol + mutant_v2))
    )
    # return layered_decision_engine(test_case, S_O, S_M, tol)

def build_fingerprints(mutant_funcs, test_cases, oracle_func, all_behavior_types):
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}

    for name, mutant_func in mutant_funcs.items():  # 第1层
        vec = np.zeros(len(all_behavior_types))
        killed_list = []

        for test_case in test_cases:  # 第2层
            killed, oracle_viol, mutant_viol = run_test(oracle_func, mutant_func, test_case)

            killed_list.append(1 if killed else 0)

            for viol in mutant_viol:  # 第3层
                if viol in all_behavior_types:
                    idx = all_behavior_types.index(viol)
                    vec[idx] += 1

        ms_per_mutant[name] = np.mean(killed_list) if killed_list else 0.0
        fingerprints[name] = np.array(killed_list, dtype=float)
        violation_map[name] = vec

    return fingerprints, ms_per_mutant, violation_map

# 调试函数：打印未命中的违规类型
def check_missing_violations(fingerprints, violation_map, mutant_funcs, test_cases, oracle_func):
    all_types = set(all_behavior_types)
    hit_types = set()
    
    for name, vec in violation_map.items():
        for i, count in enumerate(vec):
            if count > 0:
                hit_types.add(all_behavior_types[i])
    
    missing = all_types - hit_types
    if missing:
        print(f"未命中的违规类型 ({len(missing)}): {missing}")
             

    else:
        print("所有违规类型都已命中！")


# 使用示例
if __name__ == "__main__":
    test_cases = generate_lhs_samples(n=200, seed=42)
    print(f"生成了 {len(test_cases)} 个测试用例")
    # print("\n前10个用例示例:")
    # for i, case in enumerate(test_cases[:10]):
    #     print(f"  {i+1}: {case}")

    # 加载原始代码
    oracle = load_oracle()
    # if oracle:
    #     print(f"Oracle loaded: {oracle}")
    #     print(f"Test: relu(-5) = {oracle(-5)}")
    #     print(f"Test: relu([-1, 2, -3]) = {oracle([-1, 2, -3])}")
    
    # 加载所有变异体    
    mutants = load_mutants()
    print(f"\n已加载 {len(mutants)} mutants")
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutants, test_cases,oracle, all_behavior_types)

    # 定义四分类归属
    categories = {
        'Numerical Stability': [3, 4, 5, 6, 14, 15, 16],
        'Statistical Properties': [13],
        'Semantic / Logic': [7, 8, 9, 12],
        'Structural / Dimension': [0, 1, 2, 10, 11, 17]
    }   

    #region RQ1 McNemar
    a=analyze_single_operator('Relu',kill_matrix,violation_map)
    print_operator_summary(a)
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
