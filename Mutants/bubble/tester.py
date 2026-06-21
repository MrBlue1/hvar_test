import os
import sys
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import copy
from scipy.stats import qmc

mutants_dir=Path(__file__).parent
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
# ========== 行为指纹标签定义（冒泡排序专用） ==========
# 根据排序算法的行为特征定义
LABELS = [
    "正确",           # 完全正确排序
    "未排序",         # 结果未按升序排列
    "元素丢失",       # 数组长度变短（元素被删除）
    "元素增加",       # 数组长度变长（添加了额外元素）
    "类型错误",       # 返回非列表类型（如None, int等）
    "部分排序",       # 部分有序但非完全有序（如只排了一半）
    "索引越界",       # 访问数组越界异常
    "其他异常"        # 其他运行时异常
]
label2idx = {l: i for i, l in enumerate(LABELS)}

def oracle(arr: List) -> List:
    """标准答案：正确的冒泡排序（升序）"""
    if not isinstance(arr, list):
        raise TypeError("Input must be a list")
    
    n = len(arr)
    result = copy.deepcopy(arr)  # 深拷贝避免修改原数组
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result

def is_sorted(arr: List) -> bool:
    """检查列表是否已按升序排序"""
    if not isinstance(arr, list) or len(arr) < 2:
        return True
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def classify_behavior(expected: List, actual: Any, exception: Optional[Exception] = None) -> str:
    """
    对冒泡排序变异体的行为进行分类，返回 LABELS 中的字符串标签
    """
    if exception is not None:
        # 根据异常类型细分
        if isinstance(exception, (IndexError, TypeError)) and "index" in str(exception).lower():
            return "索引越界"
        else:
            return "其他异常"
    
    # 类型检查：必须是列表
    if not isinstance(actual, list):
        return "类型错误"
    
    # 空值检查
    if actual is None:
        return "类型错误"  # None视为类型错误
    
    # 长度检查
    if len(actual) < len(expected):
        return "元素丢失"
    if len(actual) > len(expected):
        return "元素增加"
    
    # 长度正确，检查内容
    if actual == expected:
        return "正确"
    
    # 检查是否有序（但可能不是预期结果，如有重复元素排序不稳定等情况）
    if is_sorted(actual):
        # 如果有序但和预期不同，可能是排序稳定性或比较逻辑问题
        # 这里简化为"未排序"（预期不匹配）
        if sorted(actual) == sorted(expected):
            return "未排序"  # 有序但和预期不同（理论上不应发生，除非oracle也错了）
        else:
            return "元素丢失"  # 内容被篡改但未改变长度
    
    # 检查是否部分有序（简单启发式：检查是否至少有一对是正确顺序）
    sorted_pairs = sum(1 for i in range(len(actual)-1) if actual[i] <= actual[i+1])
    total_pairs = len(actual) - 1
    
    if 0 < sorted_pairs < total_pairs:
        return "部分排序"
    
    return "未排序"

def load_sort_func(py_file: Path):
    """加载变异体的 bubble_sort 函数"""
    name = py_file.stem
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.bubble_sort

def lhs_samples(n: int, seed: Optional[int] = None, max_len: int = 50,
                val_range: tuple = (-100, 100)) -> List[List[int]]:
    """
    使用 LHS 原理生成 n 个 BUBBLE 测试用例。
    
    每个测试用例控制三个维度：
    1. 数组长度 [1, max_len]
    2. 排序程度 [0,1]，0=完全逆序，1=完全有序
    3. 重复元素比例 [0,0.5]，最多重复一半长度
    
    Args:
        n: 测试用例数量
        seed: 随机种子
        max_len: 数组最大长度
        val_range: 元素值范围 (min_val, max_val)
    
    Returns:
        List[List[int]]: 测试用例数组列表
    """
    if seed is not None:
        np.random.seed(seed)
    
    # --- 生成 LHS 样本 ---
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    lhs_raw = sampler.random(n)  # shape [n,3], 每列 [0,1)
    
    # 映射到实际范围
    lengths = np.floor(lhs_raw[:,0] * (max_len - 1) + 1).astype(int)
    orderness = lhs_raw[:,1]  # [0,1)
    duplicates = lhs_raw[:,2] * 0.5  # 0~0.5
    
    samples = []
    
    for i in range(n):
        length = lengths[i]
        arr = np.random.randint(val_range[0], val_range[1]+1, size=length)
        
        # 根据排序程度生成数组
        if orderness[i] < 0.33:
            # 完全随机
            arr = np.random.permutation(arr)
        elif orderness[i] < 0.66:
            # 部分有序
            mid = max(1, length // 2)
            arr[:mid] = np.sort(arr[:mid])
            arr = np.random.permutation(arr) if np.random.rand() < 0.5 else arr
        else:
            # 几乎有序
            arr = np.sort(arr)
            # 随机交换少量元素
            n_swaps = max(1, int(length * 0.1))
            for _ in range(n_swaps):
                idx1, idx2 = np.random.randint(0, length, size=2)
                arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
        
        # 插入重复元素
        n_dup = int(length * duplicates[i])
        for _ in range(n_dup):
            val = np.random.choice(arr)
            pos = np.random.randint(0, length)
            arr[pos] = val  # 替换而不是 insert，保证长度不变
        
        samples.append(arr.tolist())
    
    return samples


def run_tests(sort_func, tests: List[List]):
    """
    执行测试并返回拼接后的 one-hot 向量
    形状: [n_tests * len(LABELS)]
    """
    cov = []
    
    for test_arr in tests:
        expected = oracle(test_arr)
        actual = None
        exception = None
        
        try:
            # 注意：某些变异体可能会修改输入数组，所以传入副本
            actual = sort_func(copy.deepcopy(test_arr))
        except Exception as e:
            exception = e
        
        # 确定行为标签
        label = classify_behavior(expected, actual, exception)
        
        # 创建 one-hot 向量
        v = np.zeros(len(LABELS))
        v[label2idx[label]] = 1
        cov.append(v)
    
    return np.concatenate(cov)  # shape = [n_tests * n_labels]

def generate_coverage_vectors(n_tests: int = 20000, 
                              mutants_dir: Optional[Path] = None,
                              max_arr_len: int = 50) -> Dict[str, np.ndarray]:
    """
    生成冒泡排序变异测试的覆盖向量
    
    Args:
        MUTANT_OPERATOR_MAP: 变异体ID映射，例如 {'M01': 'ROR', 'M02': 'AOR'}
        n_tests: 测试用例数量
        mutants_dir: 变异体所在目录，默认为本文件所在目录
        max_arr_len: 测试数组最大长度
    
    Returns:
        coverage_vectors: {mid: concatenated_onehot_vector}
    """
    if mutants_dir is None:
        mutants_dir = Path(__file__).parent
    
    # 生成 LHS 测试用例
    print(f"生成 {n_tests} 个 LHS 测试用例（数组长度 1-{max_arr_len}）...")
    tests = lhs_samples(n_tests, max_len=max_arr_len)
    
    coverage_vectors = {}
    
    print("开始执行变异测试...")
    for mid in MUTANT_OPERATOR_MAP:
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            print(f"警告: 未找到 {py_file}，跳过")
            continue
            
        print(f"测试 {mid} ({MUTANT_OPERATOR_MAP[mid]})...", end=' ')
        try:
            sort_func = load_sort_func(py_file)
            coverage_vectors[mid] = run_tests(sort_func, tests)
            print(f"完成 (shape={coverage_vectors[mid].shape})")
        except Exception as e:
            print(f"加载失败: {e}")
            # 全部标记为其他异常
            v = np.zeros(n_tests * len(LABELS))
            v[label2idx["其他异常"]] = 1
            # 复制到整个向量
            for i in range(n_tests):
                v[i * len(LABELS) + label2idx["其他异常"]] = 1
            coverage_vectors[mid] = v
    
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
    # 假设 MUTANT_OPERATOR_MAP 由您提供（冒泡排序的变异体映射）
    MOCK_MAP = {
        "M01": "original",
        "M02": "ROR",  # 关系运算符变异
        "M03": "AOR",  # 算术运算符变异
        "M04": "COR",  # 条件变异
        "M05": "SDL"   # 语句删除
    }
    
    # 运行测试
    cov_vectors = generate_coverage_vectors(MOCK_MAP, n_tests=100)
    
    print("\nCoverage Vectors 摘要:")
    for mid, vec in cov_vectors.items():
        # 计算每个标签的出现次数
        n_labels = len(LABELS)
        counts = {label: 0 for label in LABELS}
        for i in range(0, len(vec), n_labels):
            label_idx = int(vec[i:i+n_labels].argmax())
            counts[LABELS[label_idx]] += 1
        
        print(f"\n{mid}:")
        for label, count in counts.items():
            if count > 0:
                print(f"  {label}: {count}次 ({count/(len(vec)/n_labels):.1%})")