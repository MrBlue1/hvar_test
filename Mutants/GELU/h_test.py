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
import threading
import concurrent.futures
import warnings
from pathlib import Path
from contextlib import contextmanager
import copy

from Mutants.mutant_ananlysis import analyze_single_operator,print_operator_summary

all_behavior_types = [
    # ========= 基础数值异常 =========
    "invalid_output",           # 无效输出
    "nan",                      # 非数值
    "inf",                      # 无穷大
    
    # ========= 类型与结构错误 =========
    "type_error",               # 类型错误
    "recursion_error",          # 递归错误（如列表自引用）
    "list_length_mismatch",     # 列表长度/内容不匹配
    
    # ========= GELU 数学特性违反 =========
    "monotonicity_violation",   # 违反单调性
    "sign_preservation_fail",   # 符号保持失败
    "asymptotic_mismatch",      # 渐近行为错误
    "range_violation",          # 输出范围异常
    "zero_symmetry_broken",     # 零点对称性违反
    
    # ========= 浮点精度错误 =========
    "precision_loss",           # 精度损失
    "erf_approximation_error",  # 误差函数近似误差过大
    "underflow",                # 下溢（新增）
    "overflow",                 # 上溢（新增）
    
    # ========= 性能与资源错误 =========
    "performance_degradation",  # 性能严重下降（新增）
    "memory_blowup",            # 内存爆炸（新增）
    
    # ========= 安全与边界错误 =========
    "data_corruption",          # 数据损坏（输入被意外修改）
    "side_effect_leak",         # 副作用泄露（如修改了全局状态）
]



# ========== 1. 测试用例生成 ==========
def generate_lhs_samples(n: int, seed: Optional[int] = None) -> List[Any]:
    if seed is not None:
        random.seed(seed)
    
    n1 = int(n * 0.4)   # 常规功能测试
    n2 = int(n * 0.4)   # 针对性故障触发
    n3 = n - n1 - n2    # 边界攻击
    
    # 策略1：常规测试（包含极端值）
    extreme_values = [0.0, 1.0, -1.0, 1e-6, -1e-6, 1e6, -1e6, 1e-10, -1e-10,
                      10.0, -10.0, 100.0, -100.0, 0.5, -0.5, 2.0, -2.0]
    normal_values = [random.uniform(-5, 5) for _ in range(max(0, n1 - len(extreme_values)))]
    strategy1 = extreme_values + normal_values
    random.shuffle(strategy1)
    strategy1 = strategy1[:n1]
    
    # 策略2：针对性违规触发
    def triggers_for_type(bt: str) -> List[Any]:
        if bt == "invalid_output": return [None, "abc"]
        if bt == "nan": return [float('nan'), [float('nan'), 1.0]]
        if bt == "inf": return [float('inf'), -float('inf')]
        if bt == "type_error": return [{"a": 1}, {1,2,3}]
        if bt == "recursion_error":
            lst = [1,2]
            lst.append(lst)
            return [lst, [lst]]
        if bt == "list_length_mismatch": return [[1,2,3], [4,5,6]]
        if bt == "monotonicity_violation": return [-1.0, 1.0]
        if bt == "sign_preservation_fail": return [-0.5, 0.5]
        if bt == "asymptotic_mismatch": return [1e10, -1e10]
        if bt == "range_violation": return [1e-100, -1e-100]
        if bt == "zero_symmetry_broken": return [0, 0.0]
        if bt == "precision_loss": return [1e-8, -1e-8]
        if bt == "erf_approximation_error": return [0.7, -0.7]
        if bt == "underflow": return [1e-308, -1e-308]
        if bt == "overflow": return [1e308, -1e308]
        if bt == "performance_degradation": return [[1]*1000, list(range(500))]
        if bt == "memory_blowup": return [[1]*10000, list(range(5000))]
        if bt == "data_corruption": return [1.0, [1,2,3]]
        if bt == "side_effect_leak": return [1.0, [1,2]]
        return [0,1]
    
    all_triggers = []
    for bt in all_behavior_types:
        all_triggers.extend(triggers_for_type(bt))
    if len(all_triggers) > n2:
        strategy2 = random.sample(all_triggers, n2)
    else:
        strategy2 = all_triggers[:]
        while len(strategy2) < n2:
            base = random.choice(strategy2)
            if isinstance(base, (int, float)):
                perturbed = base + random.uniform(-1e-6, 1e-6)
            elif isinstance(base, list):
                perturbed = base[:]
                if perturbed and isinstance(perturbed[0], (int, float)):
                    perturbed[0] += random.uniform(-1e-6, 1e-6)
                else:
                    perturbed.append(random.uniform(-1,1))
            else:
                perturbed = base
            strategy2.append(perturbed)
    
    # 策略3：边界攻击
    attack_modes = [
        lambda: float('nan'), lambda: float('inf'), lambda: -float('inf'),
        lambda: 1e-324, lambda: -1e-324, lambda: 1e308, lambda: -1e308,
        lambda: 0.0, lambda: -0.0, lambda: 1e-7, lambda: -1e-7,
        lambda: [float('inf'), 0], lambda: [float('nan'), 1],
        lambda: [0]*1000, lambda: list(range(100))
    ]
    strategy3 = [random.choice(attack_modes)() for _ in range(n3)]
    
    all_cases = strategy1 + strategy2 + strategy3
    if len(all_cases) > n:
        all_cases = random.sample(all_cases, n)
    return all_cases

# =========================================================
# 载入 Oracle & Mutants
# =========================================================

def load_oracle() -> Any:
    """
    载入mutants目录下的M00.py原始代码作为oracle
    
    Returns:
        原始gelu函数对象
    """
    mutants_dir = Path('mutants')
    m00_path = mutants_dir / 'M00.py'
    
    # 确保文件存在
    if not m00_path.exists():
        raise FileNotFoundError(f"找不到原始代码文件: {m00_path}")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("oracle_gelu", m00_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["oracle_gelu"] = module
    spec.loader.exec_module(module)
    
    # 返回gelu函数
    if hasattr(module, 'gelu'):
        return module.gelu
    else:
        raise AttributeError("M00.py中未找到gelu函数")

def load_mutants() -> Dict[str, Any]:
    """
    载入mutants目录下的所有变异体代码（M01.py, M02.py, ...）
    
    Returns:
        字典，键为变异体名称（如'M01'），值为对应的gelu函数对象
    """
    mutants_dir = Path('mutants')
    if not mutants_dir.exists():
        raise FileNotFoundError(f"找不到mutants目录: {mutants_dir}")
    
    mutants = {}
    
    # 遍历mutants目录下所有.py文件
    for py_file in sorted(mutants_dir.glob('M*.py')):
        mutant_name = py_file.stem  # 'M00', 'M01', 'M02', ...
        
        # 跳过原始代码（M00）
        if mutant_name == 'M00':
            continue
        
        try:
            # 动态导入每个变异体模块
            spec = importlib.util.spec_from_file_location(f"mutant_{mutant_name}", py_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"mutant_{mutant_name}"] = module
            spec.loader.exec_module(module)
            
            # 提取gelu函数
            if hasattr(module, 'gelu'):
                mutants[mutant_name] = module.gelu
            else:
                print(f"警告: {py_file.name} 中未找到gelu函数，已跳过")
                
        except Exception as e:
            print(f"警告: 加载 {py_file.name} 时出错: {e}，已跳过")
    
    return mutants

# ==========================
# 三层决策引擎
# ==========================

# 全局容差
DEFAULT_TOL = 1e-9


# ----------------------------------------------------------------------
# ========== 辅助函数 ==========
def map_exception_to_violation(e: Exception) -> str:
    """将异常映射到违规类型"""
    if isinstance(e, TypeError):
        return "type_error"
    elif isinstance(e, RecursionError):
        return "recursion_error"
    elif isinstance(e, ZeroDivisionError):
        return "invalid_output"
    elif isinstance(e, OverflowError):
        return "overflow"
    elif isinstance(e, MemoryError):
        return "memory_blowup"
    elif isinstance(e, TimeoutError):
        return "performance_degradation"
    else:
        return "invalid_output"

def map_warning_to_violation(warning: warnings.WarningMessage) -> Optional[str]:
    """将警告映射到违规类型"""
    msg = str(warning.message).lower()
    if "division by zero" in msg:
        return "invalid_output"
    if "overflow" in msg:
        return "overflow"
    if "underflow" in msg:
        return "underflow"
    if "precision" in msg:
        return "precision_loss"
    return None

def safe_list_compare(list1, list2, tol=1e-9, depth=0, max_depth=100):
    """
    安全地比较两个列表，避免递归深度问题
    """
    if depth > max_depth:
        return True  # 超过最大深度，认为相等
    
    if len(list1) != len(list2):
        return False
    
    for a, b in zip(list1, list2):
        if isinstance(a, list) and isinstance(b, list):
            if not safe_list_compare(a, b, tol, depth + 1, max_depth):
                return False
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not np.isclose(a, b, atol=tol):
                return False
        elif a != b:
            return False
    return True

def safe_deepcopy(obj, max_depth=100, current_depth=0):
    """
    安全地深拷贝对象，避免递归深度问题
    """
    if current_depth > max_depth:
        return obj
    
    if isinstance(obj, list):
        return [safe_deepcopy(item, max_depth, current_depth + 1) for item in obj]
    else:
        return copy.deepcopy(obj)

def detect_recursive_structure(obj, seen=None, max_depth=100):
    """
    检测对象是否包含递归结构（自引用）
    """
    if seen is None:
        seen = set()
    
    if id(obj) in seen:
        return True
    
    seen.add(id(obj))
    
    if isinstance(obj, list):
        for item in obj:
            if detect_recursive_structure(item, seen, max_depth):
                return True
    
    return False

# ----------------------------------------------------------------------
# 第1个函数：检测数学性质违规（基于输出值）
def detect_violations(test_case: Any, output: Any, tol: float = 1e-9) -> List[str]:
    """完整的违规检测 - 覆盖所有类型"""
    violations = []
    
    # === 1. 基础异常 ===
    if output is None:
        violations.append("invalid_output")
        return violations
    
    # === 2. NaN/Inf 检测 ===
    if isinstance(output, float):
        if math.isnan(output):
            violations.append("nan")
            return violations
        if math.isinf(output):
            violations.append("inf")
            return violations
    
    # === 3. 类型错误 ===
    if not isinstance(output, (int, float, list)):
        violations.append("invalid_output")
        return violations
    
    # === 4. 下溢/上溢 ===
    if isinstance(output, float):
        if 0 < abs(output) < 1e-307:
            violations.append("underflow")
        if abs(output) > 1e307:
            violations.append("overflow")
    
    # === 5. 递归错误 ===
    if isinstance(output, list):
        try:
            # 尝试检测递归结构
            if detect_recursive_structure(output):
                violations.append("recursion_error")
                return violations
            repr(output)  # 如果还有问题，repr 会触发 RecursionError
        except RecursionError:
            violations.append("recursion_error")
            return violations
    
    # === 6. 列表长度不匹配 ===
    if isinstance(output, list) and isinstance(test_case, list):
        try:
            if len(test_case) != len(output):
                violations.append("list_length_mismatch")
        except RecursionError:
            violations.append("recursion_error")
            return violations
    
    # === 7. 内存爆炸 ===
    if isinstance(output, list) and len(output) > 10000:
        violations.append("memory_blowup")
    
    # === 8. 数值输出的数学性质 ===
    if isinstance(output, (int, float)) and isinstance(test_case, (int, float)):
        # 符号保持失败
        if test_case > tol and output <= 0:
            violations.append("sign_preservation_fail")
        elif test_case < -tol and output >= 0:
            violations.append("sign_preservation_fail")
        
        # 零点对称破坏
        if abs(test_case) < tol and abs(output) > tol:
            violations.append("zero_symmetry_broken")
        
        # 范围违规
        if abs(output) > 1e100:
            violations.append("range_violation")
        
        # 渐近行为错误
        if test_case > 1e8:
            ratio = output / test_case if test_case != 0 else 0
            if ratio < 0.99 or ratio > 1.01:
                violations.append("asymptotic_mismatch")
        elif test_case < -1e8:
            if abs(output) > 1e-6:
                violations.append("asymptotic_mismatch")
        
        # 精度损失和erf近似误差
        try:
            expected = 0.5 * test_case * (1 + math.erf(test_case / math.sqrt(2)))
            if abs(output - expected) > 1e-6:
                violations.append("precision_loss")
            if 0.3 < abs(test_case) < 0.8 and abs(output - expected) > 1e-7:
                violations.append("erf_approximation_error")
        except:
            pass
    
    return violations

# ----------------------------------------------------------------------
# 第2个函数：分层决策引擎
def layered_decision_engine(test_case: Any, S_O: Any, S_M: Any, tol: float = 1e-5) -> Tuple[bool, List[str], List[str]]:
    """
    分层决策引擎：判断变异体是否被杀死
    None 值会被归类为违规类型
    """
    # 处理原始函数的 None 值
    if S_O is None:
        ov = ["invalid_output"]
    else:
        ov = detect_violations(test_case, S_O, tol)
    
    # 处理变异体函数的 None 值
    if S_M is None:
        mv = ["invalid_output"]
    else:
        mv = detect_violations(test_case, S_M, tol)
    
    # # 情况1：都有违规
    # if ov and mv:
    #     # 违规类型完全相同则未杀死，否则杀死
    #     if sorted(ov) == sorted(mv):
    #         return False, ov, mv
    #     else:
    #         return True, ov, mv
    
    # # 情况2：仅一方违规 -> 杀死
    # if ov or mv:
    #     return True, ov, mv
    
    # # 情况3：都无违规 -> 比较输出（此时 S_O 和 S_M 都不是 None）
    # # 比较列表
    # if isinstance(S_O, list) and isinstance(S_M, list):
    #     if len(S_O) != len(S_M):
    #         return True, [], []
    #     for a, b in zip(S_O, S_M):
    #         if isinstance(a, (int, float)) and isinstance(b, (int, float)):
    #             if not np.isclose(a, b, atol=tol):
    #                 return True, [], []
    #         elif a != b:
    #             return True, [], []
    #     return False, [], []
    

    return not np.allclose(S_O, S_M, atol=tol), oracle_viol, mutant_viol
    # 比较标量
    # try:
    #     if isinstance(S_O, (int, float)) and isinstance(S_M, (int, float)):
    #         if np.isclose(S_O, S_M, atol=tol):
    #             return False, [], []
    #         else:
    #             return True, [], []
    #     else:
    #         return (S_O != S_M), [], []
    # except Exception:
    #     return True, [], []

# ----------------------------------------------------------------------
# 第3个函数：运行单个测试用例（捕获异常/警告）
def run_test(
    oracle_func, 
    mutant_func, 
    test_case: Any, 
    tol: float = 1e-9, 
    timeout: float = 1.0
) -> Tuple[bool, List[str], List[str]]:
    """
    执行单个测试用例的完整测试
    
    参数:
        oracle_func: 原始代码函数
        mutant_func: 变异体函数
        test_case: 测试用例
        tol: 数值比较容差
        timeout: 超时时间（秒）
    
    返回:
        killed: 变异体是否被杀死
        oracle_viol: 原始代码的违规类型列表
        mutant_viol: 变异体的违规类型列表
    """
    
    # ========== 1. 保存原始状态（用于检测副作用和数据损坏） ==========
    # 注意：对于包含递归结构的列表，使用安全的深拷贝
    original_test_case = None
    has_recursive = False
    
    try:
        if isinstance(test_case, (int, float)):
            original_test_case = test_case
        elif isinstance(test_case, list):
            # 检测是否包含递归结构
            if detect_recursive_structure(test_case):
                has_recursive = True
                original_test_case = test_case  # 递归结构无法深拷贝，直接引用
            else:
                original_test_case = safe_deepcopy(test_case)
    except Exception:
        original_test_case = test_case
    
    # 保存模块状态（用于检测副作用）
    modules_before = set(sys.modules.keys())
    
    # 初始化结果
    S_O = None
    S_M = None
    oracle_viol = []
    mutant_viol = []
    
    # ========== 2. 辅助函数：带超时的执行 ==========
    def run_with_timeout(func, arg, timeout_sec):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, arg)
            try:
                return future.result(timeout=timeout_sec), None
            except concurrent.futures.TimeoutError:
                return None, "timeout"
            except Exception as e:
                return None, e
    
    # ========== 3. 执行原始函数 ==========
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result, error = run_with_timeout(oracle_func, test_case, timeout)
        
        if error == "timeout":
            oracle_viol.append("performance_degradation")
        elif error is not None:
            oracle_viol.append(map_exception_to_violation(error))
        else:
            S_O = result
        
        # 捕获警告
        for warn in w:
            viol = map_warning_to_violation(warn)
            if viol and viol not in oracle_viol:
                oracle_viol.append(viol)
    
    # ========== 4. 执行变异体函数 ==========
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result, error = run_with_timeout(mutant_func, test_case, timeout)
        
        if error == "timeout":
            mutant_viol.append("performance_degradation")
        elif error is not None:
            mutant_viol.append(map_exception_to_violation(error))
        else:
            S_M = result
        
        # 捕获警告
        for warn in w:
            viol = map_warning_to_violation(warn)
            if viol and viol not in mutant_viol:
                mutant_viol.append(viol)
    
    # ========== 5. 检测数据损坏 ==========
    # 跳过包含递归结构的测试用例
    if not has_recursive and original_test_case is not None:
        try:
            if isinstance(original_test_case, (int, float)):
                if test_case != original_test_case:
                    mutant_viol.append("data_corruption")
            elif isinstance(original_test_case, list):
                # 使用安全的列表比较
                if len(test_case) != len(original_test_case):
                    mutant_viol.append("data_corruption")
                elif test_case != original_test_case:
                    # 对于可能包含递归结构的列表，避免深度比较
                    if not detect_recursive_structure(test_case) and not detect_recursive_structure(original_test_case):
                        if test_case != original_test_case:
                            mutant_viol.append("data_corruption")
        except (RecursionError, RuntimeError):
            # 比较过程中出现递归错误，忽略
            pass
    
    # ========== 6. 检测副作用 ==========
    modules_after = set(sys.modules.keys())
    if modules_after - modules_before:
        mutant_viol.append("side_effect_leak")
    
    # ========== 7. 调用 detect_violations 进行数学性质检测 ==========
    # 只有当函数正常返回且有有效输出时才进行数学性质检测
    if error is None or error != "timeout":
        try:
            if S_O is not None:
                ov_math = detect_violations(test_case, S_O, tol)
                for v in ov_math:
                    if v not in oracle_viol:
                        oracle_viol.append(v)
        except (RecursionError, RuntimeError):
            oracle_viol.append("recursion_error")
    
    if error is None or error != "timeout":
        try:
            if S_M is not None:
                mv_math = detect_violations(test_case, S_M, tol)
                for v in mv_math:
                    if v not in mutant_viol:
                        mutant_viol.append(v)
        except (RecursionError, RuntimeError):
            mutant_viol.append("recursion_error")
    
    # ========== 8. 决策逻辑：判断变异体是否被杀死 ==========
    
    # 情况1：都有违规
    if oracle_viol and mutant_viol:
        # 违规类型完全相同则未杀死，否则杀死
        if sorted(oracle_viol) == sorted(mutant_viol):
            return False, oracle_viol, mutant_viol
        else:
            return True, oracle_viol, mutant_viol
    
    # 情况2：仅一方违规 -> 杀死
    if oracle_viol or mutant_viol:
        return True, oracle_viol, mutant_viol
    
    # 情况3：都无违规，但执行失败（S_O 或 S_M 为 None）
    if S_O is None:
        oracle_viol.append("invalid_output")
    if S_M is None:
        mutant_viol.append("invalid_output")
    
    if oracle_viol or mutant_viol:
        return True, oracle_viol, mutant_viol
    
    # 情况4：都无违规且都有有效输出 -> 比较输出值
    try:
        # 比较列表
        if isinstance(S_O, list) and isinstance(S_M, list):
            # 使用安全的列表比较
            if safe_list_compare(S_O, S_M, tol):
                return False, [], []
            else:
                return True, [], []
        
        # 比较标量
        if isinstance(S_O, (int, float)) and isinstance(S_M, (int, float)):
            if np.isclose(S_O, S_M, atol=tol):
                return False, [], []
            else:
                return True, [], []
        else:
            return (S_O != S_M), [], []
    except (RecursionError, RuntimeError):
        # 比较过程中出现递归错误，认为不同
        return True, [], []
    except Exception:
        return True, [], []


# ----------------------------------------------------------------------
# 第4个函数：构建指纹信息
def build_fingerprints(mutant_funcs: Dict[str, Any], test_cases: List[Any], oracle_func: Any,
                       tol: float = 1e-9) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]:
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}
    for name, m_func in mutant_funcs.items():
        killed_list = []
        vec = np.zeros(len(all_behavior_types), dtype=int)
        for tc in test_cases:
            killed, _, mut_viol = run_test(oracle_func, m_func, tc, tol)
            killed_list.append(1 if killed else 0)
            for viol in mut_viol:
                if viol in all_behavior_types:
                    idx = all_behavior_types.index(viol)
                    vec[idx] += 1
        fingerprints[name] = np.array(killed_list, dtype=float)
        ms_per_mutant[name] = np.mean(killed_list) if killed_list else 0.0
        violation_map[name] = vec
    return fingerprints, ms_per_mutant, violation_map


# 调试函数：打印未命中的违规类型
def check_missing_violations(violation_map, test_cases=None, mutant_funcs=None, oracle_func=None):
    """
    检查哪些违规类型没有被任何变异体触发
    
    参数:
        violation_map: 从build_fingerprints返回的violation_map
        test_cases, mutant_funcs, oracle_func: 保留参数用于可能的扩展调试
    
    返回:
        missing: 未命中的违规类型集合
        hit_types: 已命中的违规类型集合
    """
    all_types = set(all_behavior_types)
    hit_types = set()
    
    # 遍历所有变异体的违规统计向量
    for name, vec in violation_map.items():
        # vec 是长度为 len(all_behavior_types) 的数组
        for i, count in enumerate(vec):
            if count > 0:  # 该违规类型至少被触发一次
                hit_types.add(all_behavior_types[i])
    
    missing = all_types - hit_types
    
    # 打印统计信息
    print(f"\n=== 违规类型命中统计 ===")
    print(f"总违规类型数: {len(all_types)}")
    print(f"已命中: {len(hit_types)} 个")
    print(f"命中率: {len(hit_types)/len(all_types)*100:.1f}%")
    
    if missing:
        print(f"未命中的违规类型 ({len(missing)}):")
        for mt in sorted(missing):
            print(f"  - {mt}")
        
    else:
        print("✓ 所有违规类型都已命中！")
    
    return missing, hit_types

# 使用示例
if __name__ == "__main__":
    test_cases = generate_lhs_samples(n=50)
    print(f"生成了 {len(test_cases)} 个测试用例")

    # 加载原始代码
    oracle = load_oracle()
    
    # 加载所有变异体    
    mutants = load_mutants()
    print(f"\n已加载 {len(mutants)} 个变异体")
    
    # 构建指纹信息
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutants, test_cases, oracle)
    print("指纹构建完成",violation_map)
    # missing, hit_types = check_missing_violations(violation_map)

    # run_violation_statistc(all_behavior_types,violation_map)

    # # run_upc_calculation(all_behavior_types,violation_map)

    # # 定义四分类归属
    categories = {
        'Numerical Stability': [1, 2, 13, 14, 11, 12],     # nan, inf, underflow, overflow, precision_loss, erf_approximation_error
        'Statistical Properties': [7, 10, 8, 9, 6],        # sign_preservation_fail, zero_symmetry_broken, asymptotic_mismatch, range_violation, monotonicity_violation
        'Semantic / Logic': [0, 3, 4, 17, 18, 15, 16],    # invalid_output, type_error, recursion_error, data_corruption, side_effect_leak, performance_degradation, memory_blowup
        'Structural / Dimension': [5],                     # list_length_mismatch
    }

    #region RQ1 McNemar
    a=analyze_single_operator('Gelu',kill_matrix,violation_map,n_sample_round=10)
    print_operator_summary(a)
    #endregion

    # #region 碰撞试验
    # case1=detect_fingerprint_collisions(kill_matrix,violation_map)
    # collid_graph_t(case1,total_mutants=76, all_behavior_types=all_behavior_types,title='GELU',categories=categories)
    # print_collision_report(case1,len(mutants))
    # #endregion