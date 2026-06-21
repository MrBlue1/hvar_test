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
from Mutants.mutant_ananlysis import analyze_single_operator,print_operator_summary
from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies,plot_cases

# 更新后的行为类型列表
all_behavior_types = [
    # ========== 输入/结构违规 ==========
    "dimension_mismatch",          # 0: 内部维度不匹配
    "empty_input",                 # 1: A 或 B 为空矩阵
    "irregular_rows",              # 2: 矩阵行长度不一致
    "non_numeric_input",           # 3: 矩阵元素包含非数值
    "scalar_input",                # 4: 输入为标量而非二维列表
    "wrong_iteration_bound",       # 5: 循环边界错误导致越界或遗漏

    # ========== 数值异常 ==========
    "nan_in_output",               # 6: 输出含有 NaN
    "inf_in_output",               # 7: 输出含有 Inf 或 -Inf
    "overflow_warning",            # 8: 触发溢出警告
    "underflow_to_zero",           # 9: 极小值下溢为 0.0
    "sign_error",                  # 10: 输出符号错误
    "precision_loss",              # 11: 大数加小数导致精度严重丢失

    # ========== 数学性质违规 ==========
    "incorrect_product_value",     # 12: 任意元素计算错误
    "wrong_output_shape",          # 13: 输出形状不是 (m x p)
    "identity_violation",          # 14: A * I != A 或 I * A != A
    "zero_property_violation",     # 15: 零矩阵乘任意矩阵结果非全零
    "transpose_property_violation",# 16: (A@B)^T != B^T @ A^T

    # ========== 性能/副作用违规 ==========
    "aliasing_side_effect",        # 17: 修改了输入矩阵的元素
]

# ========== 上下文管理器：抑制输出 ==========
@contextmanager
def suppress_all_output():
    """抑制 stdout 和 stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def generate_lhs_samples(n: int, seed: Optional[int] = None,
                         max_dim: int = 10,
                         value_range: Tuple[float, float] = (-100.0, 100.0)) -> List[Tuple[Any, Any]]:
    """
    生成覆盖所有违规类型的测试用例
    """
    random.seed(seed)
    np.random.seed(seed)
    samples = []

    def add(A, B):
        samples.append((A, B))

    # 1. 正常用例 (20%)
    n_normal = max(1, int(n * 0.2))
    for _ in range(n_normal):
        m = random.randint(1, max_dim)
        k = random.randint(1, max_dim)
        p = random.randint(1, max_dim)
        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
        add(A, B)

    # 2. 针对性违规用例 (60%)
    targeted_types = [
        # 结构违规
        "dimension_mismatch", "empty_input", "irregular_rows", "non_numeric_input",
        "scalar_input", "wrong_iteration_bound",
        # 数值异常
        "nan_in_output", "inf_in_output", "overflow_warning", "underflow_to_zero",
        "sign_error", "precision_loss",
        # 数学性质
        "incorrect_product_value", "wrong_output_shape", "identity_violation",
        "zero_property_violation", "transpose_property_violation",
        # 副作用
        "aliasing_side_effect",
    ]
    
    per_type = max(3, int(n * 0.6) // len(targeted_types))

    for vtype in targeted_types:
        for _ in range(per_type):
            m, k, p = random.randint(2, max(5, max_dim)), random.randint(2, max(5, max_dim)), random.randint(2, max(5, max_dim))
            
            try:
                if vtype == "dimension_mismatch":
                    # k2 = k + random.randint(1, 3)
                    # A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    # B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k2)]
                    # add(A, B)
                    mismatch_type = random.randint(0, 4)
                    if mismatch_type == 0:
                        # A的列数 != B的行数 (多1)
                        k2 = k + random.randint(1, 3)
                        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k2)]
                    elif mismatch_type == 1:
                        # A的列数 != B的行数 (少1)
                        k2 = max(1, k - random.randint(1, 2))
                        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k2)]
                    elif mismatch_type == 2:
                        # B的行数与A的列数完全不匹配
                        k2 = k * 2
                        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k2)]
                    elif mismatch_type == 3:
                        # 1x1 与 2x2 不匹配
                        A = [[random.uniform(*value_range)]]
                        B = [[random.uniform(*value_range) for _ in range(2)] for _ in range(2)]
                    else:
                        # 2x3 与 2x3 不匹配（应该是3x?）
                        A = [[random.uniform(*value_range) for _ in range(3)] for _ in range(2)]
                        B = [[random.uniform(*value_range) for _ in range(3)] for _ in range(2)]
                    add(A, B)
                    
                elif vtype == "empty_input":
                    # if random.random() < 0.5:
                    #     add([], [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)])
                    # else:
                    #     add([[random.uniform(*value_range) for _ in range(k)] for _ in range(m)], [])
                    empty_type = random.randint(0, 6)
                    if empty_type == 0:
                        # A为空列表
                        add([], [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)])
                    elif empty_type == 1:
                        # B为空列表
                        add([[random.uniform(*value_range) for _ in range(k)] for _ in range(m)], [])
                    elif empty_type == 2:
                        # A和B都为空
                        add([], [])
                    elif empty_type == 3:
                        # A为[[]]（有一行但空）
                        add([[]], [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)])
                    elif empty_type == 4:
                        # B为[[]]
                        add([[random.uniform(*value_range) for _ in range(k)] for _ in range(m)], [[]])
                    elif empty_type == 5:
                        # A的行中有空行
                        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                        A.append([])
                        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                        add(A, B)
                    else:
                        # B的行中有空行
                        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                        B.append([])
                        add(A, B)
                        
                elif vtype == "irregular_rows":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    if m > 0:
                        irregular_idx = random.randint(0, m-1)
                        A[irregular_idx] = [random.uniform(*value_range) for _ in range(random.randint(1, max(1, k-1)))]
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "non_numeric_input":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    A[0][0] = random.choice(["not_a_number", None, "string", complex(1,2)])
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "scalar_input":
                    if random.random() < 0.5:
                        add(42.5, [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)])
                    else:
                        add([[random.uniform(*value_range) for _ in range(k)] for _ in range(m)], 42.5)
                        
                elif vtype == "wrong_iteration_bound":
                    # 使用边界值来触发循环边界错误
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    # 添加一个标记，使某些变异体更容易触发边界错误
                    add(A, B)
                    
                elif vtype == "nan_in_output":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    A[0][0] = float('nan')
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "inf_in_output":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    A[0][0] = float('inf')
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "overflow_warning":
                    # 使用极大值
                    huge_val = 1e150
                    A = [[huge_val] * k for _ in range(m)]
                    B = [[huge_val] * p for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "underflow_to_zero":
                    # 使用极小值
                    tiny_val = 1e-150
                    A = [[tiny_val] * k for _ in range(m)]
                    B = [[tiny_val] * p for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "sign_error":
                    A = [[abs(random.uniform(1, 10)) for _ in range(k)] for _ in range(m)]
                    B = [[-abs(random.uniform(1, 10)) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "precision_loss":
                    A = [[1e15] * k for _ in range(m)]
                    B = [[1e-15] * p for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "incorrect_product_value":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "wrong_output_shape":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
                elif vtype == "identity_violation":
                    dim = random.randint(2, max(5, max_dim))
                    I = [[1.0 if i==j else 0.0 for j in range(dim)] for i in range(dim)]
                    B_mat = [[random.uniform(*value_range) for _ in range(dim)] for _ in range(dim)]
                    if random.random() < 0.5:
                        add(I, B_mat)
                    else:
                        add(B_mat, I)
                        
                elif vtype == "zero_property_violation":
                    Z = [[0.0] * k for _ in range(m)]
                    B_mat = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    if random.random() < 0.5:
                        add(Z, B_mat)
                    else:
                        B_mat = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                        Z2 = [[0.0] * p for _ in range(k)]
                        add(B_mat, Z2)
                        
                elif vtype == "transpose_property_violation":
                    # 使用方阵
                    dim = random.randint(2, max(5, max_dim))
                    A = [[random.uniform(*value_range) for _ in range(dim)] for _ in range(dim)]
                    B = [[random.uniform(*value_range) for _ in range(dim)] for _ in range(dim)]
                    add(A, B)
                    
                elif vtype == "aliasing_side_effect":
                    A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                    B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                    add(A, B)
                    
            except Exception as e:
                # 如果生成失败，添加一个正常用例
                A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
                B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
                add(A, B)

    # 3. 边界用例 (20%)
    n_boundary = max(1, int(n * 0.2))
    for _ in range(n_boundary):
        mode = random.randint(0, 5)
        m = k = p = random.randint(1, max_dim)
        
        if mode == 0:  # 全零
            add([[0.0]*k for _ in range(m)], [[0.0]*p for _ in range(k)])
        elif mode == 1:  # 全负数
            add([[-1.0]*k for _ in range(m)], [[-1.0]*p for _ in range(k)])
        elif mode == 2:  # 极值混合
            add([[1e-100]*k for _ in range(m)], [[1e100]*p for _ in range(k)])
        elif mode == 3:  # 1x1矩阵
            add([[random.uniform(*value_range)]], [[random.uniform(*value_range)]])
        elif mode == 4:  # 混合NaN/Inf
            A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
            B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
            if m > 0 and k > 0:
                A[0][0] = random.choice([float('nan'), float('inf'), -float('inf')])
            add(A, B)
        elif mode == 5:  # 1xN 和 Nx1 矩阵
            A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(1)]
            B = [[random.uniform(*value_range)] for _ in range(k)]
            add(A, B)
        elif mode == 6:  # 维度不匹配边界
            # 故意生成不匹配的维度
            A = [[random.uniform(*value_range) for _ in range(3)] for _ in range(2)]
            B = [[random.uniform(*value_range) for _ in range(2)] for _ in range(3)]
            add(A, B)
        elif mode == 7:  # 空矩阵边界
            add([[]], [[random.uniform(*value_range)]])

    # 4. 确保包含特定的维度不匹配和空输入测试用例
    # 添加更多维度不匹配的用例
    dimension_mismatch_cases = [
        # A: 2x3, B: 4x2 (不匹配)
        ([[1,2,3], [4,5,6]], [[1,2], [3,4], [5,6], [7,8]]),
        # A: 3x2, B: 3x3 (不匹配)
        ([[1,2], [3,4], [5,6]], [[1,2,3], [4,5,6], [7,8,9]]),
        # A: 1x1, B: 2x2 (不匹配)
        ([[1]], [[1,2], [3,4]]),
        # A: 2x2, B: 1x1 (不匹配)
        ([[1,2], [3,4]], [[1]]),
    ]
    
    for A, B in dimension_mismatch_cases:
        if len(samples) < n:
            add(A, B)
    
    # 添加更多空输入用例
    empty_cases = [
        ([], []),
        ([[]], [[]]),
        ([[1,2], []], [[1], [2]]),
        ([[1,2]], []),
        ([], [[1,2]]),
    ]
    
    for A, B in empty_cases:
        if len(samples) < n:
            add(A, B)

    # 补足数量
    while len(samples) < n:
        m = random.randint(1, max_dim)
        k = random.randint(1, max_dim)
        p = random.randint(1, max_dim)
        A = [[random.uniform(*value_range) for _ in range(k)] for _ in range(m)]
        B = [[random.uniform(*value_range) for _ in range(p)] for _ in range(k)]
        add(A, B)

    random.shuffle(samples)
    return samples[:n]

# =========================================================
# 载入 Oracle & Mutants
# =========================================================

def load_oracle(mutants_dir: str = "mutants") -> Optional[Callable]:
    """加载原始函数"""
    path = os.path.join(mutants_dir, "M00.py")
    if not os.path.exists(path):
        return None
    
    with suppress_all_output():
        try:
            spec = importlib.util.spec_from_file_location("M00", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.matmul if hasattr(mod, "matmul") else None
        except:
            return None

def load_mutants(mutants_dir: str = "mutants", 
                 start: int = 1, 
                 end: Optional[int] = None) -> Tuple[Dict[str, Callable], Dict[str, List[str]]]:
    """
    加载变异体，返回函数字典和预存违规字典
    """
    mutant_funcs = {}
    pre_violations = {}
    mutant_path = Path(mutants_dir)
    
    if not mutant_path.exists():
        return mutant_funcs, pre_violations
    
    if end is None:
        existing_files = list(mutant_path.glob("M[0-9][0-9].py"))
        if not existing_files:
            return mutant_funcs, pre_violations
        max_num = max(int(f.stem[1:]) for f in existing_files)
        end = max_num
    
    for i in range(start, end + 1):
        name = f"M{i:02d}"
        path = mutant_path / f"{name}.py"
        
        if not path.exists():
            continue
        
        with suppress_all_output():
            try:
                # 先检查语法错误
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, str(path), 'exec')
                
                # 无语法错误，正常加载
                spec = importlib.util.spec_from_file_location(name, str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                
                if hasattr(mod, "matmul"):
                    mutant_funcs[name] = mod.matmul
                    pre_violations[name] = []
                else:
                    pre_violations[name] = ["empty_input"]
                    
            except SyntaxError as e:
                pre_violations[name] = ["syntax_error"]
                mutant_funcs[name] = None
            except Exception as e:
                error_msg = str(e).lower()
                viols = []
                if "dimension" in error_msg:
                    viols.append("dimension_mismatch")
                elif "index" in error_msg:
                    viols.append("wrong_iteration_bound")
                else:
                    viols.append("incorrect_product_value")
                pre_violations[name] = viols
                mutant_funcs[name] = None
    
    return mutant_funcs, pre_violations

# ========== 辅助函数：数值矩阵比较 ==========
def _is_close_matrix(A, B, tol=1e-8):
    if A is None or B is None:
        return False
    with np.errstate(all='ignore'):
        try:
            A_arr = np.asarray(A, dtype=np.float64)
            B_arr = np.asarray(B, dtype=np.float64)
            if A_arr.shape != B_arr.shape:
                return False
            return np.allclose(A_arr, B_arr, rtol=tol, atol=tol, equal_nan=True)
        except:
            return False

def _is_zero_matrix(M, tol=1e-8):
    if M is None:
        return False
    with np.errstate(all='ignore'):
        try:
            arr = np.asarray(M, dtype=np.float64)
            return np.all(np.abs(arr) <= tol)
        except:
            return False

def _check_identity(A, tol=1e-8):
    with np.errstate(all='ignore'):
        try:
            arr = np.asarray(A, dtype=np.float64)
            if arr.shape[0] != arr.shape[1]:
                return False
            return np.allclose(arr, np.eye(arr.shape[0]), rtol=tol, atol=tol)
        except:
            return False

def _check_overflow_possible(A, B):
    """检查是否可能发生溢出"""
    try:
        A_arr = np.asarray(A, dtype=np.float64)
        B_arr = np.asarray(B, dtype=np.float64)
        # 检查是否有大数相乘
        max_A = np.max(np.abs(A_arr))
        max_B = np.max(np.abs(B_arr))
        if max_A > 1e100 and max_B > 1e100:
            return True
        return False
    except:
        return False

def _check_underflow_possible(A, B):
    """检查是否可能发生下溢"""
    try:
        A_arr = np.asarray(A, dtype=np.float64)
        B_arr = np.asarray(B, dtype=np.float64)
        # 检查是否有小数相乘
        max_A = np.max(np.abs(A_arr[A_arr != 0])) if np.any(A_arr != 0) else 0
        max_B = np.max(np.abs(B_arr[B_arr != 0])) if np.any(B_arr != 0) else 0
        if max_A < 1e-100 and max_B < 1e-100:
            return True
        return False
    except:
        return False

# ========== 1. detect_violations ==========
def detect_violations(
    test_case: Tuple[Any, Any],
    result: Any,
    exception: Optional[Exception],
    warnings_list: List[warnings.WarningMessage],
    inputs_modified: bool = False,
    exec_time: Optional[float] = None,
    oracle_func: Optional[Callable] = None,
) -> List[str]:
    """
    根据执行结果检测所有违规类型。
    """
    violations = []
    A, B = test_case

    # ---------- 增强的空输入检测 ----------
    # 检查输入是否为空
    def is_empty_input(obj):
        if obj is None:
            return True
        if isinstance(obj, list):
            if len(obj) == 0:
                return True
            # 检查是否所有行都为空
            if all(isinstance(row, list) and len(row) == 0 for row in obj):
                return True
        return False
    
    if is_empty_input(A) or is_empty_input(B):
        violations.append("empty_input")
    
    # 检查维度不匹配（静态分析）
    try:
        if isinstance(A, list) and isinstance(B, list):
            if len(A) > 0 and len(B) > 0:
                if isinstance(A[0], list) and isinstance(B[0], list):
                    # 检查矩阵乘法的维度兼容性
                    if len(A[0]) != len(B):
                        violations.append("dimension_mismatch")
                elif not isinstance(A[0], list):
                    # A不是二维列表
                    pass
    except:
        pass


    # ---------- 结构/输入违规 ----------
    # if exception is not None:
    #     msg = str(exception).lower()
    #     e_type = type(exception).__name__.lower()
        
    #     if "dimension" in msg or "shape" in msg or "align" in msg:
    #         violations.append("dimension_mismatch")
    #     if "empty" in msg or "length" in msg:
    #         violations.append("empty_input")
    #     if "scalar" in msg or "subscriptable" in msg or ("int" in msg and "has no len" in msg):
    #         violations.append("scalar_input")
    #     if "index" in msg:
    #         if "out of range" in msg or "index" in msg:
    #             violations.append("wrong_iteration_bound")
    #     if "type" in msg and ("str" in msg or "non-numeric" in msg):
    #         violations.append("non_numeric_input")
        
    #     # 检查异常类型
    #     if "typeerror" in e_type and "scalar" not in violations:
    #         violations.append("scalar_input")
    #     if "indexerror" in e_type:
    #         violations.append("wrong_iteration_bound")
    #     if "valueerror" in e_type:
    #         if "dimension" not in violations:
    #             violations.append("dimension_mismatch")

    # 检查输入是否锯齿状或含非数值
    try:
        if isinstance(A, list) and len(A) > 0:
            row_lens = [len(row) if isinstance(row, list) else -1 for row in A]
            if len(set(row_lens)) > 1:
                violations.append("irregular_rows")
        if isinstance(B, list) and len(B) > 0:
            row_lens = [len(row) if isinstance(row, list) else -1 for row in B]
            if len(set(row_lens)) > 1:
                violations.append("irregular_rows")
    except:
        pass
    if exception is not None:
        msg = str(exception).lower()
        e_type = type(exception).__name__.lower()
        
        # 增强维度不匹配的检测
        if any(keyword in msg for keyword in ["dimension", "shape", "align", "multiply", "matmul"]):
            violations.append("dimension_mismatch")
        if any(keyword in msg for keyword in ["empty", "length", "index out of range", "pop from empty"]):
            violations.append("empty_input")
        if any(keyword in msg for keyword in ["scalar", "subscriptable", "has no len", "object is not subscriptable"]):
            violations.append("scalar_input")
        if any(keyword in msg for keyword in ["index", "out of range", "list index out of range"]):
            violations.append("wrong_iteration_bound")
        if any(keyword in msg for keyword in ["type", "str", "non-numeric", "unsupported operand"]):
            violations.append("non_numeric_input")
        
        # 根据异常类型判断
        if "valueerror" in e_type:
            if "dimension" in msg or "shape" in msg:
                violations.append("dimension_mismatch")
            elif "empty" in msg:
                violations.append("empty_input")
        elif "typeerror" in e_type:
            if "scalar" not in violations:
                violations.append("scalar_input")
        elif "indexerror" in e_type:
            violations.append("wrong_iteration_bound")

    # 检查输入维度（即使没有异常也检查）
    try:
        if isinstance(A, list) and isinstance(B, list):
            # 检查空输入
            if len(A) == 0 or len(B) == 0:
                violations.append("empty_input")
            elif len(A) > 0 and isinstance(A[0], list) and len(A[0]) == 0:
                violations.append("empty_input")
            elif len(B) > 0 and isinstance(B[0], list) and len(B[0]) == 0:
                violations.append("empty_input")
            
            # 检查维度匹配
            if (len(A) > 0 and isinstance(A[0], list) and 
                len(B) > 0 and isinstance(B, list)):
                if len(A[0]) != len(B):
                    violations.append("dimension_mismatch")
    except:
        pass

    def _has_non_numeric(mat):
        try:
            for row in mat:
                for x in row:
                    if not isinstance(x, (int, float, np.number)):
                        return True
        except:
            return False
        return False

    if _has_non_numeric(A) or _has_non_numeric(B):
        violations.append("non_numeric_input")

    # 检查输出维度错误
    if result is not None and exception is None:
        try:
            m = len(A) if isinstance(A, list) else 0
            p = len(B[0]) if isinstance(B, list) and len(B) > 0 and isinstance(B[0], list) else 0
            if not isinstance(result, list):
                violations.append("wrong_output_shape")
            elif len(result) != m:
                violations.append("wrong_output_shape")
            elif m > 0 and len(result) > 0:
                if not isinstance(result[0], list) or len(result[0]) != p:
                    violations.append("wrong_output_shape")
        except:
            pass

    # ---------- 数值异常 ----------
    if result is not None and exception is None:
        with np.errstate(all='ignore'):
            try:
                arr = np.asarray(result, dtype=np.float64)
                if np.isnan(arr).any():
                    violations.append("nan_in_output")
                if np.isinf(arr).any():
                    violations.append("inf_in_output")
                    
                # 检查是否可能发生溢出/下溢（基于输入特征）
                if _check_overflow_possible(A, B):
                    # 如果输入值很大，检查结果是否有异常大的值
                    if np.max(np.abs(arr[~np.isinf(arr)])) > 1e200 if np.any(~np.isinf(arr)) else False:
                        violations.append("overflow_warning")
                
                if _check_underflow_possible(A, B):
                    # 检查是否有非零值变成零
                    non_zero_input = (np.any(np.asarray(A) != 0) and np.any(np.asarray(B) != 0))
                    if non_zero_input and np.all(arr == 0):
                        violations.append("underflow_to_zero")
            except:
                pass

    # 检查警告信息
    for w in warnings_list:
        msg = str(w.message).lower()
        if "overflow" in msg:
            violations.append("overflow_warning")
        if "underflow" in msg:
            violations.append("underflow_to_zero")
        if "invalid" in msg or "nan" in msg:
            violations.append("nan_in_output")
        if "divide by zero" in msg:
            violations.append("inf_in_output")
        if "precision" in msg or "round" in msg:
            violations.append("precision_loss")

    # ---------- 数学性质违规 ----------
    if oracle_func is not None and result is not None and exception is None:
        with np.errstate(all='ignore'):
            try:
                oracle_result = oracle_func(A, B)
                if oracle_result is not None:
                    # 检查乘积值是否正确
                    if not _is_close_matrix(result, oracle_result, tol=1e-6):
                        # 进一步检查是否是精度损失
                        arr_o = np.asarray(oracle_result, dtype=np.float64)
                        arr_m = np.asarray(result, dtype=np.float64)
                        if arr_o.shape == arr_m.shape:
                            max_diff = np.max(np.abs(arr_o - arr_m))
                            # 检查符号错误
                            if max_diff > 1e-3:
                                mask_nonzero = (arr_o != 0) & (arr_m != 0)
                                if np.any(mask_nonzero):
                                    signs_o = np.sign(arr_o[mask_nonzero])
                                    signs_m = np.sign(arr_m[mask_nonzero])
                                    if np.any(signs_o != signs_m):
                                        violations.append("sign_error")
                                
                                # 检查精度损失
                                rel_diff = np.abs(arr_o - arr_m) / (np.abs(arr_o) + 1e-10)
                                if np.any(rel_diff > 0.1):  # 相对误差超过10%
                                    violations.append("precision_loss")
                                else:
                                    violations.append("incorrect_product_value")
                            else:
                                violations.append("incorrect_product_value")
            except Exception as e:
                pass

            # 转置性质检查
            try:
                if (isinstance(A, list) and isinstance(B, list) and
                    len(A) > 0 and len(B) > 0 and
                    isinstance(A[0], list) and isinstance(B[0], list)):
                    
                    # 使用oracle计算正确结果
                    oracle_result = oracle_func(A, B)
                    if oracle_result is not None and len(oracle_result) > 0:
                        BT = [[B[j][i] for j in range(len(B))] for i in range(len(B[0]))]
                        AT = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
                        BT_AT_oracle = oracle_func(BT, AT)
                        
                        # 计算变异体结果的转置
                        result_T = [[result[j][i] for j in range(len(result))] for i in range(len(result[0]))]
                        
                        if BT_AT_oracle is not None and not _is_close_matrix(result_T, BT_AT_oracle, tol=1e-6):
                            violations.append("transpose_property_violation")
            except Exception as e:
                pass

            # 单位矩阵性质
            try:
                if isinstance(A, list) and len(A) > 0 and isinstance(A[0], list):
                    if len(A) == len(A[0]) and _check_identity(A):
                        if not _is_close_matrix(result, B, tol=1e-6):
                            violations.append("identity_violation")
            except:
                pass
            
            try:
                if isinstance(B, list) and len(B) > 0 and isinstance(B[0], list):
                    if len(B) == len(B[0]) and _check_identity(B):
                        if not _is_close_matrix(result, A, tol=1e-6):
                            violations.append("identity_violation")
            except:
                pass

            # 零矩阵性质
            if _is_zero_matrix(A) or _is_zero_matrix(B):
                if not _is_zero_matrix(result, tol=1e-6):
                    violations.append("zero_property_violation")

    # ---------- 副作用 ----------
    if inputs_modified:
        violations.append("aliasing_side_effect")

    return list(set(violations))

def _deep_equal(obj1: Any, obj2: Any, tol: float = 1e-8) -> bool:
    """深度比较两个对象是否相等"""
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        for item1, item2 in zip(obj1, obj2):
            if not _deep_equal(item1, item2, tol):
                return False
        return True
    elif isinstance(obj1, (int, float, np.number)):
        if math.isnan(obj1) and math.isnan(obj2):
            return True
        if math.isinf(obj1) and math.isinf(obj2):
            return obj1 == obj2
        return abs(obj1 - obj2) <= tol
    else:
        return obj1 == obj2

# ========== 3. run_test ==========
def run_test(oracle_func: Callable,
             mutant_func: Optional[Callable],
             test_case: Tuple[Any, Any],
             pre_violations: List[str] = None) -> Tuple[bool, List[str], List[str]]:
    
    A, B = test_case
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)
    oracle_viol = []
    mutant_viol = pre_violations.copy() if pre_violations else []
    S_O = None
    S_M = None
    exc_O = None
    exc_M = None
    warn_O = []
    warn_M = []
    exec_time = None

    # 执行 oracle
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            S_O = oracle_func(A, B)
        except Exception as e:
            exc_O = e
        else:
            warn_O = list(w)

    # 如果变异体为 None，直接比较预存违规
    if mutant_func is None:
        oracle_viol = detect_violations(test_case, S_O, exc_O, warn_O, False, None, oracle_func)
        killed = (set(oracle_viol) != set(mutant_viol))
        return killed, oracle_viol, mutant_viol

    # 执行 mutant
    start_time = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            S_M = mutant_func(A, B)
        except Exception as e:
            exc_M = e
        else:
            warn_M = list(w)
    exec_time = time.time() - start_time

    inputs_modified = not _deep_equal(A, A_copy) or not _deep_equal(B, B_copy)

    oracle_viol = detect_violations(test_case, S_O, exc_O, warn_O, False, None, oracle_func)
    mutant_viol = detect_violations(test_case, S_M, exc_M, warn_M, inputs_modified, exec_time, oracle_func)

    # 裁决
    if not oracle_viol and not mutant_viol:
        killed = not _is_close_matrix(S_O, S_M)
    else:
        killed = (set(oracle_viol) != set(mutant_viol))
    
    return killed, oracle_viol, mutant_viol

# ========== 4. build_fingerprints ==========
def build_fingerprints(mutant_funcs: Dict[str, Optional[Callable]],
                       test_cases: List[Tuple[Any, Any]],
                       oracle_func: Callable,
                       pre_violations: Dict[str, List[str]] = None) -> Tuple[Dict[str, np.ndarray],
                                                                              Dict[str, float],
                                                                              Dict[str, np.ndarray]]:
    """构建指纹"""
    if pre_violations is None:
        pre_violations = {}
    
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}
    num_types = len(all_behavior_types)
    
    for name, mutant_func in mutant_funcs.items():
        killed_list = []
        vec = np.zeros(num_types, dtype=int)
        pre_viols = pre_violations.get(name, [])
        
        if mutant_func is None:
            for test_case in test_cases:
                killed_list.append(1)
                for viol in pre_viols:
                    if viol in all_behavior_types:
                        idx = all_behavior_types.index(viol)
                        vec[idx] += 1
        else:
            for test_case in test_cases:
                killed, oracle_viol, mutant_viol = run_test(
                    oracle_func, mutant_func, test_case, pre_viols
                )
                killed_list.append(1 if killed else 0)
                
                for viol in mutant_viol:
                    if viol in all_behavior_types:
                        idx = all_behavior_types.index(viol)
                        vec[idx] += 1
        
        fingerprints[name] = np.array(killed_list, dtype=float)
        ms_per_mutant[name] = np.mean(killed_list) if killed_list else 0.0
        violation_map[name] = vec
    
    return fingerprints, ms_per_mutant, violation_map

# 调试函数：检查违规类型覆盖率
def check_violation_coverage(violation_map):
    """检查违规类型的覆盖情况"""
    all_types = set(all_behavior_types)
    hit_types = set()
    type_counts = {vtype: 0 for vtype in all_behavior_types}
    
    for name, vec in violation_map.items():
        for i, count in enumerate(vec):
            if count > 0:
                vtype = all_behavior_types[i]
                hit_types.add(vtype)
                type_counts[vtype] += count
    
    missing = all_types - hit_types
    coverage = len(hit_types) / len(all_types) * 100
    
    print(f"\n违规类型覆盖情况:")
    print(f"命中类型数: {len(hit_types)}/{len(all_types)} ({coverage:.1f}%)")
    
    if missing:
        print(f"\n未命中的违规类型 ({len(missing)}):")
        for vtype in sorted(missing):
            print(f"  - {vtype}")
    else:
        print("\n✓ 所有违规类型都已命中！")
    
    print(f"\n各类型命中次数:")
    for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {vtype}: {count}")
    
    return hit_types, missing, coverage

# 主程序
if __name__ == "__main__":
        # 定义四分类归属
    categories = {
            'Structural / Dimension': [0, 1, 2, 4, 5, 13],  # 结构/维度相关 (6个)
            'Numerical Stability': [6, 7, 8, 9, 11],        # 数值稳定性 (5个)
            'Statistical Properties': [10, 12, 14, 15, 16], # 统计/数学性质 (5个)
            'Semantic / Logic': [3, 17],                     # 语义/逻辑 (2个)
        }

    # 生成测试用例
    print("\n1. 生成测试用例...")
    test_cases = generate_lhs_samples(n=200, seed=42)  # 增加测试用例数量
    print(f"   生成了 {len(test_cases)} 个测试用例")

    # 加载原始代码
    print("\n2. 加载原始代码...")
    oracle = load_oracle()
    if oracle is None:
        print("   错误：无法加载原始代码 M00.py")
        sys.exit(1)
    print("   ✓ 原始代码加载成功")
    
    # 加载所有变异体    
    print("\n3. 加载变异体...")
    mutants, pre_violations = load_mutants()
    print(f"   ✓ 已加载 {len(mutants)} 个变异体")
    
    # 构建指纹
    print("\n4. 执行测试并构建指纹...")
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(
        mutants, test_cases, oracle, pre_violations
    )
    shape=np.array(list(violation_map.values())).shape

    
    #region RQ1 McNemar
    a=analyze_single_operator('MatMul',kill_matrix,violation_map)
    print_operator_summary(a)
    #endregion

    #region RQ2:experiment A-C
    print('RQ2:experiment A: Metrics')
    v=compute_rq2_metrics(kill_matrix, violation_map,categories,n_fine=18)
    print(v)

    print('RQ2:experiment B: fingerprint tsne')
    save_path = os.path.join("rq2", "tsne_MatMul.png")
    plot_fingerprint_tsne(violation_map,categories,op_name='MatMul',save_path=save_path)

    print('RQ2:experiment C: 3 Cases ')
    cases=extract_case_studies(kill_matrix,violation_map,categories,n_fine=18)
    for c in cases:
        print(f"\nCase: {c['m1']} vs {c['m2']}")
        print(f"  KM pattern: {c['km_pattern'][:5]}... (same class)")
        print(f"  FP({c['m1']}): {c['fp_m1']} → {c['layer_name_m1']} (L{c['dominant_m1']})")
        print(f"  FP({c['m2']}): {c['fp_m2']} → {c['layer_name_m2']} (L{c['dominant_m2']})")    
    plot_cases()
#endregion
    

    #region 碰撞试验
    # case1=detect_fingerprint_collisions(kill_matrix,violation_map)
    # collid_graph_t(case1,total_mutants=65, all_behavior_types=all_behavior_types,title='Sigmoid',categories=categories)
    # print_collision_report(case1,len(mutants))
    #endregion