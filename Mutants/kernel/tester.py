import os
import sys
import importlib.util
import numpy as np
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =====================================================
# 基本路径与配置
# =====================================================
mutants_dir = Path(__file__).parent
ORACLE_ID = "M00"

MUTANT_OPERATOR_MAP = {
    f"M{i:02d}": "OP" for i in range(1, 61)
}

# =====================================================
# 严格警告抑制
# =====================================================
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# =====================================================
# 行为标签（保留你原来的）
# =====================================================
LABELS = [
    "正确",
    "数值溢出",
    "非对称错误",
    "归一化错误",
    "负值错误",
    "超范围错误",
    "形状错误",
    "类型错误",
    "其他异常"
]
label2idx = {l: i for i, l in enumerate(LABELS)}

EPS = 1e-10
SYMMETRY_TOL = 1e-8
DIAGONAL_TOL = 1e-6
VALUE_TOL = 1e-5

# =====================================================
# 动态加载 rbf_kernel
# =====================================================
def load_kernel_func(py_file: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")

        name = py_file.stem
        if name in sys.modules:
            del sys.modules[name]

        spec = importlib.util.spec_from_file_location(name, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod

        with np.errstate(all="ignore"):
            spec.loader.exec_module(mod)

        if not hasattr(mod, "rbf_kernel"):
            raise ImportError(f"{py_file} 中未定义 rbf_kernel")

        return mod.rbf_kernel

# =====================================================
# LHS 测试用例生成（原逻辑）
# =====================================================
def lhs_samples(n: int, seed: int = 42, n_samples: int = 8, n_features: int = 4):
    np.random.seed(seed)
    test_cases = []
    for i in range(n):
        X = np.zeros((n_samples, n_features))
        Y = np.zeros((n_samples, n_features))
        for f in range(n_features):
            centers = np.linspace(-5, 5, max(2, n_samples // 2))
            X[:, f] = np.random.choice(centers, n_samples) + np.random.uniform(-0.5, 0.5, n_samples)
            Y[:, f] = np.random.choice(centers, n_samples) + np.random.uniform(-0.5, 0.5, n_samples)
        gamma = 10 ** np.random.uniform(-3, 3)
        eps = 10 ** np.random.uniform(-12, -4)
        det = i % 10
        if det < 3:
            test_cases.append((X, None, gamma, eps))
        elif det < 8:
            test_cases.append((X, X, gamma, eps))
        else:
            test_cases.append((X, Y, gamma, eps))
    # 边界测试
    boundary_cases = [
        (np.zeros((n_samples, n_features)), None, 1.0, 1e-8),
        (np.ones((n_samples, n_features)), None, 1.0, 1e-8),
        (np.array([[0,0],[10,0],[0,10],[10,10]]), None, 1e3, 1e-8),
        (np.random.randn(n_samples, n_features), None, 1e-3, 1e-8),
        (np.random.randn(3, n_features), np.random.randn(5, n_features), 1.0, 1e-8),
        (np.full((n_samples, n_features), 1e154), None, 1.0, 1e-8),
        (np.full((n_samples, n_features), -1e154), None, 1.0, 1e-8),
    ]
    insert_indices = np.linspace(0, n-1, len(boundary_cases), dtype=int)
    for idx, case in zip(insert_indices, boundary_cases):
        test_cases[idx] = case
    return test_cases

# =====================================================
# 数值问题检查
# =====================================================
def check_numerical_issues(arr):
    if arr is None:
        return False, False
    return np.any(np.isinf(arr)), np.any(np.isnan(arr))

# =====================================================
# 行为分类（保持你原来的，用于覆盖向量）
# =====================================================
def classify_behavior(expected, actual, X, Y, exception=None) -> str:

    if exception is not None:
        if isinstance(exception, TypeError):
            return "类型错误"
        if isinstance(exception, (ValueError, IndexError)):
            return "形状错误"
        return "其他异常"

    if not isinstance(actual, np.ndarray):
        return "类型错误"

    has_inf, has_nan = check_numerical_issues(actual)
    if has_inf or has_nan:
        return "数值溢出"

    n_x = X.shape[0]
    n_y = Y.shape[0] if Y is not None else n_x
    if actual.shape != (n_x, n_y):
        return "形状错误"

    if np.any(actual < -EPS):
        return "负值错误"

    if np.any(actual > 1.0 + VALUE_TOL):
        return "超范围错误"

    if Y is None or Y is X or np.array_equal(X, Y):
        if not np.allclose(actual, actual.T, atol=SYMMETRY_TOL):
            return "非对称错误"
        diag = np.diag(actual)
        if not np.allclose(diag, 1.0, atol=DIAGONAL_TOL):
            return "归一化错误"

    return "正确"

# =====================================================
# 运行测试，生成覆盖向量（仍保留）
# =====================================================
def run_tests(kernel_func, oracle_kernel, tests):
    cov = []

    for X, Y, gamma, eps in tests:
        exception = None
        actual = None

        try:
            with np.errstate(all="ignore"):
                actual = kernel_func(X, Y, gamma, eps)
        except Exception as e:
            exception = e

        label = classify_behavior(None, actual, X, Y, exception)
        v = np.zeros(len(LABELS))
        v[label2idx[label]] = 1
        cov.append(v)

    return np.concatenate(cov)

# =====================================================
# 覆盖向量生成（完整保留）
# =====================================================
def generate_coverage_vectors(n_tests: int,
                              mutants_dir: Optional[Path] = None,
                              seed: int = 42) -> Dict[str, np.ndarray]:
    if mutants_dir is None:
        mutants_dir = Path(__file__).parent
    tests = lhs_samples(n_tests, seed=seed)
    oracle_kernel = load_kernel_func(mutants_dir / f"{ORACLE_ID}.py")
    
    coverage_vectors = {}

    for mid in MUTANT_OPERATOR_MAP:
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            continue

        kernel_func = load_kernel_func(py_file)
        vec = run_tests(kernel_func, oracle_kernel, tests)
        coverage_vectors[mid] = vec

    return coverage_vectors

# =====================================================
# M00 行为差异判定（核心）
# =====================================================
def behavior_diff(ref, mut, atol=1e-6, rtol=1e-5) -> bool:

    if ref is None or mut is None:
        return True

    if not isinstance(ref, np.ndarray) or not isinstance(mut, np.ndarray):
        return True

    if ref.shape != mut.shape:
        return True

    if np.any(np.isnan(ref)) != np.any(np.isnan(mut)):
        return True

    if np.any(np.isinf(ref)) != np.any(np.isinf(mut)):
        return True

    return not np.allclose(ref, mut, atol=atol, rtol=rtol)

# =====================================================
# 单个变异体是否被杀死（以 M00 为 Oracle）
# =====================================================
def is_mutant_killed(oracle_kernel, mutant_kernel, tests) -> bool:

    for X, Y, gamma, eps in tests:
        try:
            with np.errstate(all="ignore"):
                ref = oracle_kernel(X, Y, gamma, eps)
                mut = mutant_kernel(X, Y, gamma, eps)
        except Exception:
            return True

        if behavior_diff(ref, mut):
            return True

    return False

# =====================================================
# Mutation Score（全体）
# =====================================================
def compute_mutation_score(coverage_vectors: Dict[str, np.ndarray]):
    """
    基于 coverage_vectors 计算 Mutation Score
    """
    killed = 0
    total = len(coverage_vectors)

    for vec in coverage_vectors.values():
        # 判断该变异体是否被杀死
        n_labels = len(LABELS)
        n_tests = vec.size // n_labels
        bv = vec.reshape(n_tests, n_labels)
        correct_idx = label2idx["正确"]

        all_correct = np.all(bv[:, correct_idx] == 1)
        if not all_correct:
            killed += 1

    return {
        "total_mutants": total,
        "killed_mutants": killed,
        "mutation_score": killed / total if total > 0 else 0.0
    }


# =====================================================
# Reduced Mutation Score
# =====================================================
def compute_reduced_mutation_score(coverage_vectors: Dict[str, np.ndarray],
                                   dict_represent: Dict[str, str]):
    """
    基于约简代表变异体的 Mutation Score
    dict_represent: {cluster_0: M01, cluster_1: M21, ...}
    """
    killed = 0
    total = len(dict_represent)

    for mid in dict_represent.values():
        if mid not in coverage_vectors:
            continue

        vec = coverage_vectors[mid]
        n_labels = len(LABELS)
        n_tests = vec.size // n_labels
        bv = vec.reshape(n_tests, n_labels)
        correct_idx = label2idx["正确"]

        all_correct = np.all(bv[:, correct_idx] == 1)
        if not all_correct:
            killed += 1

    return {
        "clusters": total,
        "killed_representatives": killed,
        "reduced_mutation_score": killed / total if total > 0 else 0.0
    }


# =====================================================
# Test-Averaged Mutation Score
# =====================================================
def compute_test_averaged_ms(coverage_vectors: Dict[str, np.ndarray]):
    """
    计算 Test-Averaged Mutation Score
    基于 coverage_vectors，每个测试用例能杀死的变异体比例
    """
    n_labels = len(LABELS)
    correct_idx = label2idx["正确"]

    # 从任意一个 mutant 推断 n_tests
    any_vec = next(iter(coverage_vectors.values()))
    if any_vec.size % n_labels != 0:
        raise ValueError("Behavior vector size not divisible by label count")
    n_tests = any_vec.size // n_labels

    killed_per_test = np.zeros(n_tests)

    for vec in coverage_vectors.values():
        if vec.size != any_vec.size:
            raise ValueError("Inconsistent behavior vector sizes across mutants")
        bv = vec.reshape(n_tests, n_labels)
        # 如果不是正确，就算该变异体被该测试用例杀死
        killed_per_test += (bv[:, correct_idx] == 0).astype(int)

    killed_ratio = killed_per_test / len(coverage_vectors)

    return {
        "n_tests": n_tests,
        "mean_test_ms": float(np.mean(killed_ratio)),
        "std_test_ms": float(np.std(killed_ratio)),
        "min_test_ms": float(np.min(killed_ratio)),
        "max_test_ms": float(np.max(killed_ratio)),
    }

# =====================================================
# 簇内部杀死率分析 / 等价变异体识别
# =====================================================
def analyze_cluster_kill_rate(coverage_vectors: Dict[str, np.ndarray],
                              dict_represent: Dict[str, str]):
    """
    输出每个簇内部成员杀死率统计
    dict_represent: {cluster_0: M01, cluster_1: M21, ...}
    """
    n_labels = len(LABELS)
    correct_idx = label2idx["正确"]
    
    cluster_stats = {}
    
    for cluster_name, rep_mid in dict_represent.items():
        # 收集属于该簇的所有成员
        members = [mid for mid in coverage_vectors.keys() if mid.startswith(rep_mid[:3])]
        member_stats = {}
        
        for mid in members:
            vec = coverage_vectors[mid]
            n_tests = vec.size // n_labels
            bv = vec.reshape(n_tests, n_labels)
            killed = np.any(bv[:, correct_idx] == 0)
            member_stats[mid] = killed
        
        killed_count = sum(member_stats.values())
        total_count = len(member_stats)
        kill_rate = killed_count / total_count if total_count > 0 else 0.0
        
        cluster_stats[cluster_name] = {
            "members": member_stats,
            "killed_count": killed_count,
            "total_count": total_count,
            "kill_rate": kill_rate
        }
    
    # 输出报告
    print("\n簇内部杀死率分析:")
    for cluster_name, stats in cluster_stats.items():
        print(f"{cluster_name}: {stats['killed_count']}/{stats['total_count']} killed | kill_rate={stats['kill_rate']:.2f}")
        low_kill_members = [mid for mid, k in stats['members'].items() if not k]
        if low_kill_members:
            print(f"  潜在等价变异体: {low_kill_members}")
    
    return cluster_stats

# =====================================================
# 主入口
# =====================================================
if __name__ == "__main__":

    print("生成 LHS 测试用例...")
    coverage_vectors=generate_coverage_vectors(200)
    full_ms = compute_mutation_score(coverage_vectors)
    
    tams = compute_test_averaged_ms(coverage_vectors)

    print("Mutation Score:", full_ms)

    print("Test-Averaged Mutation Score:", tams)


