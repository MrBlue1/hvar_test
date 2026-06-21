import os
import sys
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import copy
from scipy.stats import entropy, qmc
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ========== 原有配置（完全保留） ==========
mutants_dir = Path(__file__).parent
MUTANT_OPERATOR_MAP = {
    "M01": "ROR", "M02": "ROR", "M03": "ROR", "M04": "ROR", "M05": "ROR",
    "M06": "ROR", "M07": "ROR", "M08": "ROR", "M09": "ROR", "M10": "ROR",
    "M11": "ROR", "M12": "ROR", "M13": "AOR", "M14": "AOR", "M15": "AOR",
    "M16": "AOR", "M17": "AOR", "M18": "AOR", "M19": "AOR", "M20": "AOR",
    "M21": "AOR", "M22": "AOR", "M23": "LOR", "M24": "LOR", "M25": "LOR",
    "M26": "LOR", "M27": "LOR", "M28": "LOR", "M29": "LOR", "M30": "LOR",
    "M31": "COR", "M32": "COR", "M33": "COR", "M34": "COR", "M35": "COR",
    "M36": "COR", "M37": "UOI", "M38": "UOI", "M39": "UOI", "M40": "UOI",
    "M41": "UOI", "M42": "UOI", "M43": "UOI", "M44": "SDL", "M45": "SDL",
    "M46": "SDL", "M47": "SDL", "M48": "SDL", "M49": "SDL", "M50": "SDL",
    "M51": "ABS", "M52": "ABS", "M53": "ABS", "M54": "ABS", "M55": "ABS",
    "M56": "ABS", "M57": "ABS", "M58": "ABS", "M59": "ABS", "M60": "ABS",
}

LABELS = [
    "正确", "未排序", "元素丢失", "元素增加", 
    "类型错误", "部分排序", "索引越界", "其他异常"
]
label2idx = {l: i for i, l in enumerate(LABELS)}

# ========== 原有核心函数（完全保留） ==========
def oracle(arr: List) -> List:
    """标准答案：正确的冒泡排序（升序）"""
    if not isinstance(arr, list):
        raise TypeError("Input must be a list")
    n = len(arr)
    result = copy.deepcopy(arr)
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
    """对冒泡排序变异体的行为进行分类"""
    if exception is not None:
        if isinstance(exception, (IndexError, TypeError)) and "index" in str(exception).lower():
            return "索引越界"
        else:
            return "其他异常"
    
    if not isinstance(actual, list):
        return "类型错误"
    if actual is None:
        return "类型错误"
    if len(actual) < len(expected):
        return "元素丢失"
    if len(actual) > len(expected):
        return "元素增加"
    if actual == expected:
        return "正确"
    if is_sorted(actual):
        if sorted(actual) == sorted(expected):
            return "未排序"
        else:
            return "元素丢失"
    
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
    """使用 LHS 原理生成 n 个 BUBBLE 测试用例"""
    if seed is not None:
        np.random.seed(seed)
    
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    lhs_raw = sampler.random(n)
    
    lengths = np.floor(lhs_raw[:,0] * (max_len - 1) + 1).astype(int)
    orderness = lhs_raw[:,1]
    duplicates = lhs_raw[:,2] * 0.5
    
    samples = []
    for i in range(n):
        length = lengths[i]
        arr = np.random.randint(val_range[0], val_range[1]+1, size=length)
        
        if orderness[i] < 0.33:
            arr = np.random.permutation(arr)
        elif orderness[i] < 0.66:
            mid = max(1, length // 2)
            arr[:mid] = np.sort(arr[:mid])
            arr = np.random.permutation(arr) if np.random.rand() < 0.5 else arr
        else:
            arr = np.sort(arr)
            n_swaps = max(1, int(length * 0.1))
            for _ in range(n_swaps):
                idx1, idx2 = np.random.randint(0, length, size=2)
                arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
        
        n_dup = int(length * duplicates[i])
        for _ in range(n_dup):
            val = np.random.choice(arr)
            pos = np.random.randint(0, length)
            arr[pos] = val
        samples.append(arr.tolist())
    return samples

def run_tests(sort_func, tests: List[List]):
    """执行测试并返回拼接后的 one-hot 向量"""
    cov = []
    for test_arr in tests:
        expected = oracle(test_arr)
        actual = None
        exception = None
        try:
            actual = sort_func(copy.deepcopy(test_arr))
        except Exception as e:
            exception = e
        
        label = classify_behavior(expected, actual, exception)
        v = np.zeros(len(LABELS))
        v[label2idx[label]] = 1
        cov.append(v)
    return np.concatenate(cov)

def generate_coverage_vectors(n_tests: int = 20000, 
                              mutants_dir: Optional[Path] = None,
                              max_arr_len: int = 50) -> Dict[str, np.ndarray]:
    """生成冒泡排序变异测试的覆盖向量"""
    if mutants_dir is None:
        mutants_dir = Path(__file__).parent
    
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
            v = np.zeros(n_tests * len(LABELS))
            for i in range(n_tests):
                v[i * len(LABELS) + label2idx["其他异常"]] = 1
            coverage_vectors[mid] = v
    
    print("所有测试完成")
    return coverage_vectors

def is_mutant_killed(behavior_vector: np.ndarray) -> bool:
    """单个变异体是否被'杀死'"""
    n_labels = len(LABELS)
    if behavior_vector.size % n_labels != 0:
        raise ValueError("Behavior vector size not divisible by label count")
    n_tests = behavior_vector.size // n_labels
    bv = behavior_vector.reshape(n_tests, n_labels)
    correct_idx = label2idx["正确"]
    all_correct = np.all(bv[:, correct_idx] == 1)
    return not all_correct

def compute_mutation_score(coverage_vectors: Dict[str, np.ndarray]):
    """Mutation Score（全体变异体）"""
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

def compute_reduced_mutation_score(coverage_vectors: Dict[str, np.ndarray], dict_represent: Dict[str, str]):
    """基于约简代表变异体的 Mutation Score"""
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

# ========== 新增：数据解析与诊断模块 ==========
def extract_kill_matrix_and_profiles(coverage_vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    【新增】从coverage_vectors提取杀死矩阵和行为指纹
    Returns:
        kill_matrix: (n_mutants, n_tests) 二值矩阵
        behavior_profiles: (n_mutants, 8) 错误类型分布
        mutant_ids: 变异体ID列表（保持顺序）
    """
    mutant_ids = sorted(coverage_vectors.keys())
    n_mutants = len(mutant_ids)
    vec_shape = coverage_vectors[mutant_ids[0]].shape[0]
    n_tests = vec_shape // len(LABELS)
    
    kill_matrix = np.zeros((n_mutants, n_tests), dtype=int)
    behavior_profiles = np.zeros((n_mutants, len(LABELS)))
    correct_idx = label2idx["正确"]
    
    for i, mid in enumerate(mutant_ids):
        vec = coverage_vectors[mid]
        tests_behavior = vec.reshape(n_tests, len(LABELS))
        is_killed_per_test = (tests_behavior.argmax(axis=1) != correct_idx).astype(int)
        kill_matrix[i] = is_killed_per_test
        behavior_profiles[i] = tests_behavior.mean(axis=0)
    
    return kill_matrix, behavior_profiles, mutant_ids

def five_minute_diagnosis(kill_matrix: np.ndarray, behavior_profiles: np.ndarray) -> Dict:
    """
    【新增】5分钟快速诊断
    判断当前数据是否适合进行"行为指纹增强聚类"
    """
    n_mutants, n_tests = kill_matrix.shape
    print(f"\n{'='*60}")
    print(f"【5分钟诊断报告】样本: {n_mutants}变异体 × {n_tests}测试用例")
    print(f"{'='*60}")
    
    # 诊断A：测试集完备性
    print("\n【诊断A】测试集完备性检测")
    sample_size = min(50, n_mutants)
    indices = np.random.choice(n_mutants, sample_size, replace=False)
    jaccard_sims = []
    
    for i, j in combinations(indices, 2):
        intersection = np.sum((kill_matrix[i] == 1) & (kill_matrix[j] == 1))
        union = np.sum((kill_matrix[i] == 1) | (kill_matrix[j] == 1))
        if union > 0:
            jaccard_sims.append(intersection / union)
    
    avg_jaccard = np.mean(jaccard_sims) if jaccard_sims else 0
    
    if avg_jaccard < 0.15:
        risk_a = "HIGH"
        print(f"  ⚠️ 风险: {risk_a} - 测试集过强(Jaccard={avg_jaccard:.3f})，指纹增益有限")
    elif avg_jaccard < 0.3:
        risk_a = "MEDIUM"
        print(f"  ⚠️ 风险: {risk_a} - 区分度较好(Jaccard={avg_jaccard:.3f})")
    else:
        risk_a = "LOW"
        print(f"  ✅ 风险: {risk_a} - 存在混淆(Jaccard={avg_jaccard:.3f})，需要指纹辅助")
    
    # 诊断B：模态相关性
    print("\n【诊断B】行为指纹与杀死矩阵相关性")
    kill_labels = (kill_matrix.sum(axis=1) > 0).astype(int)
    dominant_errors = behavior_profiles.argmax(axis=1)
    mi = mutual_info_score(dominant_errors, kill_labels)
    profile_variance = np.mean(np.var(behavior_profiles, axis=0))
    
    print(f"  互信息: {mi:.3f}, 方差: {profile_variance:.3f}")
    
    if mi > 0.7 or profile_variance < 0.01:
        risk_b = "HIGH"
        print(f"  ⚠️ 风险: {risk_b} - 指纹与杀死绑定，无独立信息")
    elif mi > 0.4:
        risk_b = "MEDIUM"
        print(f"  ⚠️ 风险: {risk_b} - 存在一定相关")
    else:
        risk_b = "LOW"
        print(f"  ✅ 风险: {risk_b} - 指纹提供互补信息")
    
    # 诊断C：错误多样性
    print("\n【诊断C】错误类型多样性")
    error_counts = np.bincount(dominant_errors, minlength=len(LABELS))
    unique_errors = np.sum(error_counts > 0)
    probs = error_counts / np.sum(error_counts)
    error_entropy = entropy(probs, base=2) / np.log2(len(LABELS))
    
    print(f"  出现类型: {unique_errors}/8, 熵: {error_entropy:.3f}")
    
    if unique_errors <= 2 or error_entropy < 0.3:
        risk_c = "HIGH"
        print(f"  ⚠️ 风险: {risk_c} - 类型单一，8类分类无意义")
    elif unique_errors <= 4:
        risk_c = "MEDIUM"
        print(f"  ⚠️ 风险: {risk_c} - 类型有限")
    else:
        risk_c = "LOW"
        print(f"  ✅ 风险: {risk_c} - 类型丰富")
    
    # 综合决策
    risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    total_score = risk_map[risk_a] + risk_map[risk_b] + risk_map[risk_c]
    
    print(f"\n{'='*60}")
    if total_score >= 5:
        decision = "STOP"
        print(f"【决策】{total_score}/6 ({decision}) - 建议转向'严重性分级'或'自适应框架'")
    elif total_score >= 3:
        decision = "CAUTION"
        print(f"【决策】{total_score}/6 ({decision}) - 谨慎进行，重点对比稳定性差异")
    else:
        decision = "GO"
        print(f"【决策】{total_score}/6 ({decision}) - 全速前进")
    print(f"{'='*60}")
    
    return {
        'risk_a': risk_a, 'risk_b': risk_b, 'risk_c': risk_c,
        'total_score': total_score, 'decision': decision,
        'metrics': {'avg_jaccard': avg_jaccard, 'mi': mi, 'error_entropy': error_entropy}
    }

def cluster_with_behavior_constraint(kill_matrix: np.ndarray, 
                                    behavior_profiles: np.ndarray,
                                    mutant_ids: List[str],
                                    k: Optional[int] = None) -> Dict[str, List[str]]:
    """
    【新增】基于行为指纹约束的层次聚类
    策略：先按主导错误类型分组，再在组内基于杀死矩阵聚类
    """
    dominant_errors = behavior_profiles.argmax(axis=1)
    n_mutants = len(mutant_ids)
    if k is None:
        k = max(2, n_mutants // 6)
    
    representatives = {}
    cluster_id = 0
    
    # 按错误类型分组
    error_groups = defaultdict(list)
    for i, mid in enumerate(mutant_ids):
        error_type = dominant_errors[i]
        error_groups[error_type].append((i, mid))
    
    print(f"\n行为预分组：{len(error_groups)}个错误类型组")
    
    for error_type, members in error_groups.items():
        if len(members) == 1:
            idx, mid = members[0]
            representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [mid]
            cluster_id += 1
        else:
            indices = [m[0] for m in members]
            mids = [m[1] for m in members]
            sub_kill_matrix = kill_matrix[indices]
            
            # 简单策略：如果组内差异小，选1个代表；否则选2个
            if len(members) <= 3:
                kill_counts = sub_kill_matrix.sum(axis=1)
                best_idx = np.argmax(kill_counts)
                representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [mids[best_idx]]
                cluster_id += 1
            else:
                # 分为2个子簇
                sub_k = min(2, len(members) // 2)
                kmeans = KMeans(n_clusters=sub_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sub_kill_matrix)
                
                for sub_id in range(sub_k):
                    mask = (labels == sub_id)
                    cluster_mids = [mids[j] for j in range(len(mids)) if mask[j]]
                    if cluster_mids:
                        sub_kills = sub_kill_matrix[mask].sum(axis=1)
                        best_idx = np.argmax(sub_kills)
                        representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [cluster_mids[best_idx]]
                        cluster_id += 1
    
    return representatives

def evaluate_reduction(coverage_vectors: Dict[str, np.ndarray],
                      representatives: Dict[str, List[str]],
                      mutant_ids: List[str]) -> Dict:
    """
    【新增】评估约简质量（包含FTRR指标）
    """
    n_original = len(mutant_ids)
    n_reduced = len(representatives)
    
    full_ms = compute_mutation_score(coverage_vectors)
    reduced_ms = compute_reduced_mutation_score(coverage_vectors, 
                                               {k: v[0] for k, v in representatives.items()})
    
    # 计算FTRR（Fault Type Retention Rate）
    correct_idx = label2idx["正确"]
    original_error_types = set()
    for mid in mutant_ids:
        vec = coverage_vectors[mid].reshape(-1, len(LABELS))
        dominant = np.argmax(vec.mean(axis=0))
        if dominant != correct_idx:
            original_error_types.add(dominant)
    
    rep_error_types = set()
    for cluster, mids in representatives.items():
        for mid in mids:
            vec = coverage_vectors[mid].reshape(-1, len(LABELS))
            dominant = np.argmax(vec.mean(axis=0))
            if dominant != correct_idx:
                rep_error_types.add(dominant)
    
    ftrr = len(rep_error_types) / len(original_error_types) if original_error_types else 1.0
    
    return {
        'original_ms': full_ms['mutation_score'],
        'reduced_ms': reduced_ms['reduced_mutation_score'],
        'reduction_ratio': 1 - n_reduced/n_original,
        'ftrr': ftrr,
        'representatives': representatives
    }

# 对照组：纯杀死矩阵聚类（无行为约束）
def baseline_cluster(kill_matrix, mutant_ids, k=7):
    """无行为约束的K-Means"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(kill_matrix)
    
    representatives = {}
    for i in range(k):
        mask = (labels == i)
        cluster_indices = np.where(mask)[0]
        # 选杀死最多的
        kill_counts = kill_matrix[cluster_indices].sum(axis=1)
        best_idx = cluster_indices[np.argmax(kill_counts)]
        representatives[f"cluster_{i}"] = [mutant_ids[best_idx]]
    
    return representatives


# ========== 主流程 ==========
def main():
    """
    【新增】完整流程：生成数据 -> 5分钟诊断 -> 聚类约简 -> 评估
    """
    # 配置
    N_TESTS = 2000  # 可以改为20000测试你的上限
    MAX_ARR_LEN = 50
    
    # 步骤1: 生成数据（如果已有coverage_vectors，可以加载并跳过此步）
    print("步骤1: 生成覆盖向量...")
    coverage_vectors = generate_coverage_vectors(n_tests=N_TESTS, max_arr_len=MAX_ARR_LEN)
    
    # 步骤2: 解析
    print("\n步骤2: 解析数据...")
    kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(coverage_vectors)
    
    # 步骤3: 5分钟诊断（关键！）
    diagnosis = five_minute_diagnosis(kill_matrix, behavior_profiles)
    
    # 如果STOP，提前退出
    if diagnosis['decision'] == "STOP":
        print("\n⚠️ 诊断建议停止，请考虑：")
        print("  1. 减少测试用例数量（从2000降到200-500）")
        print("  2. 更换更复杂的被测程序（如快速排序、二叉搜索）")
        print("  3. 或改为'严重性分级'研究（不聚类，而是排序）")
        return diagnosis
    
    # 步骤4: 聚类约简
    print("\n步骤4: 执行行为约束聚类...")
    representatives = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids)
    
    # 步骤5: 评估
    evaluation = evaluate_reduction(coverage_vectors, representatives, mutant_ids)
    
    baseline_reps = baseline_cluster(kill_matrix, mutant_ids, k=7)
    baseline_eval = evaluate_reduction(coverage_vectors, baseline_reps, mutant_ids)

    

    # 报告
    print(f"\n{'='*60}")
    print("【最终评估报告】")
    print(f"{'='*60}")
    print(f"原始MS: {evaluation['original_ms']:.2%} ({compute_mutation_score(coverage_vectors)['killed_mutants']}/60)")
    print(f"约简后MS: {evaluation['reduced_ms']:.2%} ({evaluation['reduced_ms']*len(representatives):.0f}/{len(representatives)})")
    print(f"约简比例: {evaluation['reduction_ratio']:.1%} (从60降到{len(representatives)})")
    print(f"故障类型保留率(FTRR): {evaluation['ftrr']:.1%}")
    print(f"\n选出的代表 ({len(representatives)}个):")
    for cluster, mids in evaluation['representatives'].items():
        print(f"  {cluster}: {mids[0]}")
    
    print(f"对照组 FTRR: {baseline_eval['ftrr']:.1%}")
    print(f"实验组 FTRR: {evaluation['ftrr']:.1%}")

    return {
        'diagnosis': diagnosis,
        'evaluation': evaluation,
        'coverage_vectors': coverage_vectors
    }




if __name__ == "__main__":
    results = main()
    