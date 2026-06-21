import os
import sys
import importlib.util
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


mutants_dir = Path(__file__).parent
MUTANT_OPERATOR_MAP = {
    "M01": "ROR",
    "M02": "ROR",
    "M03": "ROR",
    "M04": "ROR",
    "M05": "ROR",
    "M06": "ROR",
    "M07": "ROR",
    "M08": "ROR",
    "M09": "ROR",
    "M10": "ROR",
    "M11": "ROR",
    "M12": "ROR",
    "M13": "AOR",
    "M14": "AOR",
    "M15": "AOR",
    "M16": "AOR",
    "M17": "AOR",
    "M18": "AOR",
    "M19": "AOR",
    "M20": "AOR",
    "M21": "AOR",
    "M22": "AOR",
    "M23": "LOR",
    "M24": "LOR",
    "M25": "LOR",
    "M26": "LOR",
    "M27": "LOR",
    "M28": "LOR",
    "M29": "LOR",
    "M30": "LOR",
    "M31": "COR",
    "M32": "COR",
    "M33": "COR",
    "M34": "COR",
    "M35": "COR",
    "M36": "COR",
    "M37": "UOI",
    "M38": "UOI",
    "M39": "UOI",
    "M40": "UOI",
    "M41": "UOI",
    "M42": "UOI",
    "M43": "UOI",
    "M44": "SDL",
    "M45": "SDL",
    "M46": "SDL",
    "M47": "SDL",
    "M48": "SDL",
    "M49": "SDL",
    "M50": "SDL",
    "M51": "ABS",
    "M52": "ABS",
    "M53": "ABS",
    "M54": "ABS",
    "M55": "ABS",
    "M56": "ABS",
    "M57": "ABS",
    "M58": "ABS",
    "M59": "ABS",
    "M60": "ABS",
}


# ========== 行为指纹标签定义 ==========
# 根据二次方程求解器的行为特征定义
LABELS = ["正确", "除零错误", "运算错误", "类型错误", "根数量错误", "空值错误", "其他异常"]
label2idx = {l: i for i, l in enumerate(LABELS)}

def oracle(a: float, b: float, c: float):
    """标准答案"""
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return None
        return -c / b
    d = b * b - 4 * a * c
    if d > 1e-9:
        s = math.sqrt(d)
        return ((-b + s) / (2 * a), (-b - s) / (2 * a))
    elif abs(d) <= 1e-9:
        return -b / (2 * a)
    return None

def classify_behavior(expected, actual, exception=None) -> str:
    """
    对行为进行分类，返回 LABELS 中的字符串标签
    """
    if exception is not None:
        # 根据异常类型细分
        if isinstance(exception, ZeroDivisionError):
            return "除零错误"
        elif isinstance(exception, (ValueError, ArithmeticError)):
            # ValueError: math domain error (sqrt of negative), etc.
            return "运算错误"
        else:
            return "其他异常"
    
    # 空值检查：应该返回数值但返回了None，或反之
    if actual is None and expected is not None:
        return "空值错误"
    if actual is not None and expected is None:
        return "空值错误"
    
    if actual is None and expected is None:
        # 两者都是None，表示都识别为无解/无穷解，属于正确
        return "正确"
    
    # 类型检查：tuple vs float
    if type(actual) != type(expected):
        return "类型错误"
    
    # 对于元组（双根情况），检查长度（根数量）
    if isinstance(expected, tuple):
        if len(expected) != len(actual):
            return "根数量错误"
        # 检查具体值是否相等
        if not all(abs(e - a) < 1e-6 for e, a in zip(expected, actual)):
            return "运算错误"  # 类型对、数量对，但值算错了
    
    # 单根（float）数值检查
    if isinstance(expected, (int, float)):
        if abs(expected - actual) > 1e-6:
            return "运算错误"
    
    return "正确"

def load_solve_func(py_file: Path):
    """加载变异体的 solve_quadratic 函数"""
    name = py_file.stem
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.solve_quadratic

def lhs_samples(n: int, seed: Optional[int] = None) -> List[Tuple]:
    """生成 n 个 LHS 测试用例 [(a,b,c), ...]"""
    if seed:
        np.random.seed(seed)
    samples = []
    # 对每个维度分别进行分层采样
    strata = np.linspace(0, 1, n, endpoint=False)
    for i in range(3):
        jitter = np.random.uniform(0, 1/n, n)
        dim_samples = strata + jitter
        np.random.shuffle(dim_samples)
        samples.append(dim_samples * 20 - 10)  # 映射到 [-10, 10]
    
    # 组合成 (a,b,c) 列表
    return [(samples[0][i], samples[1][i], samples[2][i]) for i in range(n)]

def run_tests(solve_func, tests: List[Tuple]):
    """
    执行测试并返回拼接后的 one-hot 向量
    形状: [n_tests * len(LABELS)]
    """
    cov = []
    for a, b, c in tests:
        expected = oracle(a, b, c)
        actual = None
        exception = None
        
        try:
            actual = solve_func(a, b, c)
        except Exception as e:
            exception = e
        
        # 确定行为标签
        label = classify_behavior(expected, actual, exception)
        
        # 创建 one-hot 向量
        v = np.zeros(len(LABELS))
        v[label2idx[label]] = 1
        cov.append(v)
    
    return np.concatenate(cov)  # shape = [n_tests * n_labels]

def generate_coverage_vectors(n_tests: int = 20000) -> Dict[str, np.ndarray]:
    """
    生成覆盖向量
    
    Args:
        MUTANT_OPERATOR_MAP: 变异体ID映射，例如 {'M01': 'desc', 'M02': 'desc'}
        n_tests: 测试用例数量
        mutants_dir: 变异体所在目录，默认为本文件所在目录
    
    Returns:
        coverage_vectors: {mid: concatenated_onehot_vector}
    """
    
    # 生成 LHS 测试用例
    print(f"生成 {n_tests} 个 LHS 测试用例...")
    tests = lhs_samples(n_tests)
    
    coverage_vectors = {}
    
    print("开始执行变异测试,MUTANT_OPERATOR_MAP...",MUTANT_OPERATOR_MAP)
    for mid in MUTANT_OPERATOR_MAP:
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            print(f"警告: 未找到 {py_file}，跳过")
            continue
            
        print(f"测试 {mid}...", end=' ')
        solve_func = load_solve_func(py_file)
        coverage_vectors[mid] = run_tests(solve_func, tests)
        print("完成")
    
    print("所有测试完成")
    return coverage_vectors

# 单个变异体是否被“杀死”
def is_mutant_killed(behavior_vector: np.ndarray) -> bool:
    n_labels = len(LABELS)

    if behavior_vector.size % n_labels != 0:
        raise ValueError("Behavior vector size not divisible by label count")

    n_tests = behavior_vector.size // n_labels
    bv = behavior_vector.reshape(n_tests, n_labels)

    correct_idx = label2idx["正确"]

    all_correct = np.all(bv[:, correct_idx] == 1)
    return not all_correct

# Mutation Score（全体变异体）
def compute_mutation_score(coverage_vectors: Dict[str, np.ndarray]):
    killed = 0
    total = len(coverage_vectors)

    for vec in coverage_vectors.values():
        if is_mutant_killed(vec):
            killed += 1

    return {
        "total_mutants": total,
        "killed_mutants": killed,
        "mutation_score": killed / total if total > 0 else 0.0
    }

# 约简集 Mutation Score
def compute_reduced_mutation_score(coverage_vectors: Dict[str, np.ndarray],
                                   dict_represent: Dict[str, str]):
    """
    基于约简代表变异体的 Mutation Score
    dict_represent: {cluster_0: M01, cluster_1: M21, ...}
    """
    killed = 0
    total = len(dict_represent)

    for cluster, mid in dict_represent.items():
        if mid not in coverage_vectors:
            continue
        if is_mutant_killed(coverage_vectors[mid]):
            killed += 1

    score = killed / total if total > 0 else 0.0

    return {
        "clusters": total,
        "killed_representatives": killed,
        "reduced_mutation_score": score
    }

# Test-averaged Mutation Score（稳健性）
# 每个 test 能杀死多少比例的变异体
def compute_test_averaged_ms(coverage_vectors: Dict[str, np.ndarray]):
    n_labels = len(LABELS)
    correct_idx = label2idx["正确"]

    # ===== 从任意一个 mutant 推断 n_tests =====
    any_vec = next(iter(coverage_vectors.values()))

    if any_vec.size % n_labels != 0:
        raise ValueError("Behavior vector size not divisible by label count")

    n_tests = any_vec.size // n_labels

    killed_per_test = np.zeros(n_tests)

    for vec in coverage_vectors.values():
        if vec.size != any_vec.size:
            raise ValueError("Inconsistent behavior vector sizes across mutants")

        bv = vec.reshape(n_tests, n_labels)
        killed_per_test += (bv[:, correct_idx] == 0).astype(int)

    killed_ratio = killed_per_test / len(coverage_vectors)

    return {
        "n_tests": n_tests,
        "mean_test_ms": float(np.mean(killed_ratio)),
        "std_test_ms": float(np.std(killed_ratio)),
        "min_test_ms": float(np.min(killed_ratio)),
        "max_test_ms": float(np.max(killed_ratio)),
    }


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例：假设 MUTANT_OPERATOR_MAP 由您提供
    MOCK_MAP = {
        "M01": "original",
        "M02": "boundary_mutation", 
        "M03": "operator_mutation"
    }
    
    # 假设这些文件存在
    cov_vectors = generate_coverage_vectors(MOCK_MAP, n_tests=100)
    
    print("\nCoverage Vectors:")
    for mid, vec in cov_vectors.items():
        print(f"{mid}: shape={vec.shape}, sum={vec.sum()}, "
              f"correct_rate={(1-vec[0::len(LABELS)].mean()):.2%}")