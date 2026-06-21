import math

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
    # 1. 运行时错误 / 非数值输出
    "nan",                    # 输出 NaN
    "inf",                    # 输出 ±inf
    "invalid_output",         # None 或非数值类型

    # 2. 数值稳定性问题
    "overflow_warning",       # 触发溢出警告 (exp 过大)
    "underflow_precision",    # 小输入导致精度丢失（exp(-x) 过小）
    "large_input_instability",# |x| > 50 时行为异常

    # 3. 输出值域违规
    "out_of_range",           # 输出不在 [0, 1] 内（允许微小误差）
    "equal_zero",             # 输出精确为 0 (或接近 0)
    "equal_one",              # 输出精确为 1 (或接近 1)

    # 4. 函数性质破坏
    "zero_input_error",       # x=0 时输出不等于 0.5
    "symmetry_violation",     # f(-x) ≠ 1 - f(x)
    "monotonicity_violation", # 单调性破坏（非严格递增）

    # 5. 符号 / 方向错误
    "sign_error",             # 输出符号与预期相反 (x>0 却 s<0.5)

    # 6. 梯度 / 平滑性异常
    "flat_region",            # 输出在非饱和区几乎不变（梯度消失）
    "discontinuity",          # 不连续（阶跃跳变）

    # 7. 结构 / 实现缺陷
    "formula_error",          # 使用了完全错误的公式
    "branch_error",           # 分支逻辑错误（如条件判断错误）
    "precision_loss_extreme", # 极端小输入时输出退化至 0 或 0.5

    # # 8. 变异体等价性标记（分析时填充）
    # "equivalent_mutant"
]

def generate_lhs_samples(n: int, seed: Optional[int] = None) -> List[Tuple[float]]:
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    
    # 强制包含 inf 和 -inf
    samples.append((float('inf'),))
    samples.append((float('-inf'),))
    
    # 策略1：正常区域（25%）
    n1 = int(n * 0.25)
    xs1 = np.linspace(-10, 10, n1)
    xs1 += np.random.uniform(-0.3, 0.3, size=n1)
    samples.extend([(float(x),) for x in xs1])
    
    # 策略2：边界/极端值（40%）
    n2 = int(n * 0.40)
    extreme_points = [
        0.0, 0.0, 0.0,
        1e-12, -1e-12, 1e-10, -1e-10,
        20, -20, 30, -30, 50, -50,
        100, -100, 200, -200, 500, -500,
        1000, -1000, 10000, -10000,  # 极大值可能触发inf
    ]
    
    for point in extreme_points:
        if len(samples) < n1 + n2 + 2:  # +2 for inf values
            samples.append((float(point),))
    
    # 补足数量
    while len(samples) < n1 + n2 + 2:
        samples.append((np.random.uniform(-10000, 10000),))
    
    # 策略3：特殊值（剩余部分）
    n3 = n - len(samples)
    specials = []
    
    for _ in range(n3):
        choice = np.random.choice([
            'nan', 'inf', '-inf',
            'very_large_positive', 'very_large_negative',
            'small', 'precise'
        ])
        
        if choice == 'nan':
            specials.append((float('nan'),))
        elif choice == 'inf':
            specials.append((float('inf'),))
        elif choice == '-inf':
            specials.append((float('-inf'),))
        elif choice == 'very_large_positive':
            specials.append((np.random.uniform(1000, 10000),))
        elif choice == 'very_large_negative':
            specials.append((np.random.uniform(-10000, -1000),))
        elif choice == 'small':
            specials.append((np.random.uniform(1e-15, 1e-12),))
        else:
            specials.append((np.random.uniform(-1e-12, 1e-12),))
    
    samples.extend(specials)
    
    # 打乱
    np.random.shuffle(samples)
    return samples[:n]


# =========================================================
# 载入 Oracle & Mutants
# =========================================================

def load_oracle():
    """
    加载原始代码 M00.py 中的 sigmoid 函数
    """
    try:
        path = os.path.join("mutants", "M00.py")
        spec = importlib.util.spec_from_file_location("M00", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "sigmoid"):
            raise AttributeError("M00.py does not contain 'sigmoid'")

        return mod.sigmoid

    except Exception as e:
        print(f"[WARN] Failed to load Oracle: {e}")
        return None
    
def load_mutants(start=1, end=67):
    """
    加载 mutants/M01.py ~ mutants/M65.py 的变异体

    返回:
        dict { 'M01': sigmoid_func, ... }
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

            # ✅ 强制检查 sigmoid 函数
            if not hasattr(mod, "sigmoid"):
                print(f"[WARN] {name} has no 'sigmoid' function")
                continue

            mutant_funcs[name] = mod.sigmoid

        except Exception as e:
            print(f"[WARN] Failed to load {name}: {e}")

    return mutant_funcs




# ==========================
# 三层决策引擎
# ==========================


# =========================
# 1️⃣ detect_violations
# =========================
def detect_violations(x, s, tol=1e-7):
    """
    x: 输入值（单个浮点数）
    s: 输出值（单个浮点数或None）
    tol: 数值容差
    """
    viols = []
    
    # 检查输入是否为inf
    if isinstance(x, float) and math.isinf(x):
        # 输入就是inf，这可能触发某些变异体的异常行为
        pass
    
    # 类型检查
    if s is None:
        viols.append("invalid_output")
        return viols
    
    # 如果s是列表或其他类型，转换为基本类型
    if not isinstance(s, (int, float, type(None))):
        try:
            if hasattr(s, '__len__') and len(s) == 1:
                s = s[0]
            else:
                viols.append("invalid_output")
                return viols
        except:
            viols.append("invalid_output")
            return viols
    
    try:
        # ========= 基础数值 =========
        if isinstance(s, float):
            if math.isnan(s):
                viols.append("nan")
            if math.isinf(s):
                viols.append("inf")
                # 如果是inf，仍然继续检查其他违规
                # 但不检查范围相关的违规
        
        # ========= 输出范围（仅当不是inf时检查）=========
        if not (isinstance(s, float) and math.isinf(s)):
            if isinstance(s, (int, float)) and not (-tol <= s <= 1 + tol):
                viols.append("out_of_range")
            
            if isinstance(s, float) and abs(s) < tol:
                viols.append("equal_zero")
            
            if isinstance(s, float) and abs(s - 1) < tol:
                viols.append("equal_one")
            
            # ========= 边界检查 =========
            if abs(x) < tol and not math.isinf(x):
                if abs(s - 0.5) > tol:
                    viols.append("zero_input_error")
            
            # ========= 符号错误 =========
            if x > tol and not math.isinf(x):
                if isinstance(s, float) and s < 0.5 - tol:
                    viols.append("sign_error")
            if x < -tol and not math.isinf(x):
                if isinstance(s, float) and s > 0.5 + tol:
                    viols.append("sign_error")
        
        # ========= 数值稳定性 =========
        if abs(x) > 100 and not math.isinf(x):
            if isinstance(s, float) and not math.isinf(s):
                if s < 1e-12 or s > 1 - 1e-12:
                    viols.append("large_input_instability")
        
        if abs(x) > 30 and not math.isinf(x):
            if isinstance(s, float) and not math.isinf(s):
                if s == 0.0:
                    viols.append("underflow_precision")
                if s == 1.0:
                    viols.append("overflow_warning")
        
        # ========= 结构错误 =========
        if isinstance(s, float) and not math.isinf(s):
            if s < -0.5 or s > 1.5:
                viols.append("formula_error")
        
        # ========= 分支错误 =========
        if abs(x) < 1e-8 and not math.isinf(x):
            if isinstance(s, float) and not math.isinf(s):
                if abs(s - 0.5) > 1e-3:
                    viols.append("branch_error")
        
        # ========= 精度丢失 =========
        if 0 < abs(x) < 1e-10 and not math.isinf(x):
            if isinstance(s, float) and not math.isinf(s):
                if s == 0.0 or s == 1.0 or s == 0.5:
                    viols.append("precision_loss_extreme")
        
        # ========= 平坦区域 =========
        if isinstance(s, float) and not math.isinf(s):
            if abs(x) > 10:
                if (x > 10 and abs(s - 1.0) < 1e-6) or (x < -10 and abs(s) < 1e-6):
                    viols.append("flat_region")
            
    except Exception as e:
        viols.append("invalid_output")
    
    return list(set(viols))


# =========================
# 2️⃣ layered_decision_engine
# =========================
def layered_decision_engine(test_case, S_O, S_M, tol=1e-7):
    x = test_case[0]

    oracle_viol = detect_violations(x, S_O, tol)
    mutant_viol = detect_violations(x, S_M, tol)

    return  not np.isclose(S_O, S_M, atol=tol, equal_nan=True)
    # 情况1：都正常
    if len(oracle_viol) == 0 and len(mutant_viol) == 0:
        if not np.isclose(S_O, S_M, atol=tol, equal_nan=True):
            return True, oracle_viol, mutant_viol
        return False, oracle_viol, mutant_viol

    # 情况2：一方违规
    if (len(oracle_viol) == 0) != (len(mutant_viol) == 0):
        return True, oracle_viol, mutant_viol

    # 情况3：违规不同
    if set(oracle_viol) != set(mutant_viol):
        return True, oracle_viol, mutant_viol

    return False, oracle_viol, mutant_viol


# =========================
# 3️⃣ run_test
# =========================
def run_test(oracle_func, mutant_func, test_case, tol=1e-7):
    # 确保test_case是单个值
    if isinstance(test_case, (list, tuple)):
        x = test_case[0]
    else:
        x = test_case

    S_O, S_M = None, None
    oracle_viol, mutant_viol = [], []

    # ===== 执行 Oracle =====
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            S_O = oracle_func(x)
            # 确保S_O是单个数值
            if isinstance(S_O, (list, tuple, np.ndarray)):
                S_O = float(S_O[0]) if len(S_O) > 0 else None
        except Exception:
            S_O = None
            oracle_viol.append("invalid_output")

        # 检查警告
        for warn in w:
            msg = str(warn.message).lower()
            if "overflow" in msg:
                oracle_viol.append("overflow_warning")
            if "underflow" in msg:
                oracle_viol.append("underflow_precision")

    # ===== 执行 Mutant =====
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            S_M = mutant_func(x)
            # 确保S_M是单个数值
            if isinstance(S_M, (list, tuple, np.ndarray)):
                S_M = float(S_M[0]) if len(S_M) > 0 else None
        except Exception:
            S_M = None
            mutant_viol.append("invalid_output")

        for warn in w:
            msg = str(warn.message).lower()
            if "overflow" in msg:
                mutant_viol.append("overflow_warning")
            if "underflow" in msg:
                mutant_viol.append("underflow_precision")

    # ===== 类型转换 =====
    if S_O is not None and not isinstance(S_O, float):
        try:
            S_O = float(S_O)
        except:
            S_O = None
            oracle_viol.append("invalid_output")
    
    if S_M is not None and not isinstance(S_M, float):
        try:
            S_M = float(S_M)
        except:
            S_M = None
            mutant_viol.append("invalid_output")

    # ===== 调用违规检测 =====
    oracle_viol.extend(detect_violations(x, S_O, tol))
    mutant_viol.extend(detect_violations(x, S_M, tol))

    # 去重
    oracle_viol = list(set(oracle_viol))
    mutant_viol = list(set(mutant_viol))

    # ===== 决策逻辑 =====
    # 情况1：两边都有违规
    if oracle_viol and mutant_viol:
        if set(oracle_viol) != set(mutant_viol):
            return True, oracle_viol, mutant_viol
        return False, oracle_viol, mutant_viol

    # 情况2：一方有违规
    if (not oracle_viol and mutant_viol) or (oracle_viol and not mutant_viol):
        return True, oracle_viol, mutant_viol

    # 情况3：都正常，比较数值
    try:
        if S_O is None or S_M is None:
            return True, oracle_viol, mutant_viol
        if not np.isclose(S_O, S_M, atol=tol, equal_nan=True):
            return True, oracle_viol, mutant_viol
    except:
        return True, oracle_viol, mutant_viol

    return False, oracle_viol, mutant_viol

# =========================
# 4️⃣ build_fingerprints
# =========================
def build_fingerprints(mutant_funcs, test_cases, oracle_func, tol=1e-7):
    """
    构建变异体指纹，包含跨用例的对称性和单调性检测
    """
    # ---------- 第一步：单用例执行与违规检测 ----------
    mutant_outputs = {}      # name -> list of outputs
    mutant_viols_per_case = {}  # name -> list of viol lists
    killed_matrix = {}       # name -> list of 0/1

    print("第一阶段：单用例测试...")
    for name, mutant_func in mutant_funcs.items():
        outputs = []
        viols_list = []
        killed_list = []
        
        for tc in test_cases:
            # 运行测试
            killed, oracle_v, mutant_v = run_test(oracle_func, mutant_func, tc, tol)
            killed_list.append(1 if killed else 0)
            viols_list.append(mutant_v)
            
            # 捕获原始输出值（用于对称性检查）
            x = tc[0] if isinstance(tc, (list, tuple)) else tc
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s_m = mutant_func(x)
                    # 确保输出是单个数值
                    if isinstance(s_m, (list, tuple, np.ndarray)):
                        s_m = float(s_m[0]) if len(s_m) > 0 else None
                    elif s_m is not None:
                        s_m = float(s_m) if isinstance(s_m, (int, float)) else None
            except:
                s_m = None
            outputs.append(s_m)
        
        mutant_outputs[name] = outputs
        mutant_viols_per_case[name] = viols_list
        killed_matrix[name] = killed_list
        
        if len(mutant_funcs) <= 10 or name in ['M01', 'M02', 'M03']:
            print(f"  {name}: 完成单用例测试")

    # ---------- 第二步：跨用例违规检测 ----------
    print("\n第二阶段：跨用例违规检测（对称性、单调性）...")
    
    # 确保tol是数值类型
    tol_value = float(tol) if not isinstance(tol, (list, tuple, dict)) else 1e-7
    
    for name in mutant_funcs:
        outputs = mutant_outputs[name]
        
        # 构建输入值到索引的映射
        x_to_output = {}
        x_to_indices = {}
        
        for idx, tc in enumerate(test_cases):
            x = tc[0] if isinstance(tc, (list, tuple)) else tc
            s = outputs[idx]
            
            # 只保存有效的数值输出
            if s is not None and isinstance(s, (int, float)) and not math.isnan(s) and not math.isinf(s):
                if x not in x_to_output:
                    x_to_output[x] = s
                    x_to_indices[x] = []
                x_to_indices[x].append(idx)
        
        # 对称性检查
        symmetry_count = 0
        for x, s_pos in list(x_to_output.items()):
            neg_x = -x
            if neg_x in x_to_output:
                s_neg = x_to_output[neg_x]
                
                try:
                    # 检查对称性：f(x) + f(-x) 应该等于 1
                    if abs(float(s_pos) + float(s_neg) - 1.0) > tol_value * 100:
                        symmetry_count += 1
                        # 标记所有相关的测试用例
                        for idx in x_to_indices.get(x, []):
                            if "symmetry_violation" not in mutant_viols_per_case[name][idx]:
                                mutant_viols_per_case[name][idx].append("symmetry_violation")
                            killed_matrix[name][idx] = 1
                        
                        for idx in x_to_indices.get(neg_x, []):
                            if "symmetry_violation" not in mutant_viols_per_case[name][idx]:
                                mutant_viols_per_case[name][idx].append("symmetry_violation")
                            killed_matrix[name][idx] = 1
                except (TypeError, ValueError) as e:
                    # 忽略类型转换错误
                    pass
        
        # ===== 不连续性检查 =====
        discontinuity_count = 0
        # 对测试用例按 x 排序后，检查相邻点输出跳变是否过大（超过阈值）
        sorted_points = []
        for idx, tc in enumerate(test_cases):
            x = tc[0] if isinstance(tc, (list, tuple)) else tc
            s = outputs[idx]
            if s is not None and isinstance(s, (int, float)) and not math.isnan(s) and not math.isinf(s):
                sorted_points.append((float(x), float(s), idx))
        sorted_points.sort(key=lambda p: p[0])
        
        for i in range(len(sorted_points) - 1):
            x1, s1, idx1 = sorted_points[i]
            x2, s2, idx2 = sorted_points[i + 1]
            # 如果输入差距很小（< 1e-4）但输出跳变很大（> 0.5），认为不连续
            if abs(x1 - x2) < 1e-4 and abs(s1 - s2) > 0.5:
                discontinuity_count += 1
                if "discontinuity" not in mutant_viols_per_case[name][idx1]:
                    mutant_viols_per_case[name][idx1].append("discontinuity")
                if "discontinuity" not in mutant_viols_per_case[name][idx2]:
                    mutant_viols_per_case[name][idx2].append("discontinuity")
                killed_matrix[name][idx1] = 1
                killed_matrix[name][idx2] = 1



        # 单调性检查
        # 收集所有有效的(x, output, index)三元组
        valid_points = []
        for idx, tc in enumerate(test_cases):
            x = tc[0] if isinstance(tc, (list, tuple)) else tc
            s = outputs[idx]
            if s is not None and isinstance(s, (int, float)) and not math.isnan(s) and not math.isinf(s):
                valid_points.append((float(x), float(s), idx))
        
        # 按x排序
        valid_points.sort(key=lambda p: p[0])
        
        monotonicity_count = 0
        for i in range(len(valid_points) - 1):
            x1, s1, idx1 = valid_points[i]
            x2, s2, idx2 = valid_points[i + 1]
            
            # 跳过相同的x值
            if abs(x1 - x2) < tol_value:
                continue
            
            # Sigmoid应该是单调递增的
            if x1 < x2 and s1 > s2 + tol_value:
                monotonicity_count += 1
                if "monotonicity_violation" not in mutant_viols_per_case[name][idx1]:
                    mutant_viols_per_case[name][idx1].append("monotonicity_violation")
                if "monotonicity_violation" not in mutant_viols_per_case[name][idx2]:
                    mutant_viols_per_case[name][idx2].append("monotonicity_violation")
                killed_matrix[name][idx1] = 1
                killed_matrix[name][idx2] = 1
        
        if symmetry_count > 0 or monotonicity_count > 0:
            print(f"  {name}: 发现 {symmetry_count} 个对称性违规, {monotonicity_count} 个单调性违规")

    # ---------- 第三步：构建最终返回值 ----------
    print("\n第三阶段：构建指纹向量...")
    fingerprints = {}
    violation_map = {}
    ms_per_mutant = {}

    for name in mutant_funcs:
        # 构建指纹向量
        killed_array = np.array(killed_matrix[name], dtype=float)
        fingerprints[name] = killed_array
        ms_per_mutant[name] = np.mean(killed_array) if len(killed_array) > 0 else 0.0
        
        # 统计违规类型
        vec = np.zeros(len(all_behavior_types))
        for viols in mutant_viols_per_case[name]:
            for viol in viols:
                if viol in all_behavior_types:
                    idx = all_behavior_types.index(viol)
                    vec[idx] += 1
        violation_map[name] = vec

    print(f"\n完成！共处理 {len(mutant_funcs)} 个变异体")
    return fingerprints, ms_per_mutant, violation_map

# =========================
# 标记等价变异体
# =========================
# def mark_equivalent_mutants(fingerprints, violation_map, ms_per_mutant, threshold=0.01):
#     """
#     标记等价变异体（那些与原始代码行为几乎完全相同的变异体）
#     """
#     equivalent_count = 0
    
#     for name in fingerprints:
#         ms = ms_per_mutant[name]
        
#         # 如果变异体几乎没有被杀死（MS < threshold），标记为等价变异体
#         if ms < threshold:
#             # 在violation_map中添加equivalent_mutant标记
#             if 'equivalent_mutant' in all_behavior_types:
#                 idx = all_behavior_types.index('equivalent_mutant')
#                 violation_map[name][idx] += 1
#             equivalent_count += 1
#             print(f"  标记 {name} 为等价变异体 (MS={ms:.4f})")
    
#     print(f"\n共标记 {equivalent_count} 个等价变异体")
#     return violation_map



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
    # 定义四分类归属
    categories = {
        'Numerical Stability': [3, 4, 5, 13, 17],           
        # overflow_warning, underflow_precision, large_input_instability, flat_region, precision_loss_extreme
        
        'Statistical Properties': [6, 7, 8, 9],             
        # out_of_range, equal_zero, equal_one, zero_input_error
        
        'Semantic / Logic': [10, 11, 12, 15],               
        # symmetry_violation, monotonicity_violation, sign_error, formula_error
        
        'Structural / Dimension': [0, 1, 2, 14, 16]         
        # nan, inf, invalid_output, discontinuity, branch_error
    }
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
    mutants = load_mutants(1,63)
    print(f"\n已加载 {len(mutants)} mutants")
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutants, test_cases,oracle)


#region experiment RQ1
    # plot_km_fp_confusion_heatmap(kill_matrix,violation_map)
    # matrix=plot_km_layer_heatmap(kill_matrix,violation_map,categories) #这个有问题
    # print(matrix)
#endregion
    
#region RQ1 significance test
    a=analyze_single_operator('Sigmoid',kill_matrix,violation_map)
    print_operator_summary(a)
#endregion
    
#region RQ2:experiment A-C
    print('RQ2:experiment A: Metrics')
    v=compute_rq2_metrics(kill_matrix, violation_map,categories)
    print(v)

    # print('RQ2:experiment B: fingerprint tsne')
    # plot_fingerprint_tsne(violation_map,categories,save_path='rq2\tsne.png')

    print('RQ2:experiment C: 3 Cases ')
    cases=extract_case_studies(kill_matrix,violation_map,categories)
    for c in cases:
        print(f"\nCase: {c['m1']} vs {c['m2']}")
        print(f"  KM pattern: {c['km_pattern'][:5]}... (same class)")
        print(f"  FP({c['m1']}): {c['fp_m1']} → {c['layer_name_m1']} (L{c['dominant_m1']})")
        print(f"  FP({c['m2']}): {c['fp_m2']} → {c['layer_name_m2']} (L{c['dominant_m2']})")    
    plot_cases()
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



    