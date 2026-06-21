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

LABELS = ["Invalid", "Equilateral", "Isosceles", "Scalene"]
label2idx = {l:i for i,l in enumerate(LABELS)}

def oracle(a: int, b: int, c: int) -> str:
    """
    Oracle函数：判断三角形的正确分类
    根据三角形边长返回正确的分类标签
    """
    # 检查是否为有效的三角形边长
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    
    # 检查三角不等式
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    
    # 判断三角形类型
    if a == b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"


def generate_lhs_triangle_tests(n=20000, low=1, high=50, seed=42):
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    sample = sampler.random(n)
    scaled = qmc.scale(sample, [low]*3, [high]*3)
    return scaled.astype(int)

lhs_tests = generate_lhs_triangle_tests()
# print('lhs测试集: ',lhs_tests)


def load_triangle_func(py_path):
    spec = importlib.util.spec_from_file_location("triangle_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triangle


def run_tests(triangle_func, tests):
    cov = []
    for a, b, c in tests:
        try:            
            out = triangle_func(a, b, c)
        except Exception:
            print('错误输入：',a,b,c)
            out = "Invalid"
        if out==None :
            out="Invalid"
        v = np.zeros(len(LABELS))
        v[label2idx[out]] = 1
        cov.append(v)
    return np.concatenate(cov)  # shape = [n_tests * 4]


def generate_coverage_vectors(n_tests: int = 20000) -> Dict[str, np.ndarray]:
    coverage_vectors = {}
    for mid in MUTANT_OPERATOR_MAP:
        py_file = mutants_dir / f"{mid}.py"
        triangle_func = load_triangle_func(py_file)
        coverage_vectors[mid] = run_tests(triangle_func, lhs_tests)
    return coverage_vectors


def is_mutant_killed(behavior_vector: np.ndarray) -> bool:
    """
    判断单个变异体是否被"杀死"
    通过与Oracle的预期结果比较，如果存在任何不一致则视为被杀死
    """
    n_labels = len(LABELS)

    if behavior_vector.size % n_labels != 0:
        raise ValueError("Behavior vector size not divisible by label count")

    n_tests = behavior_vector.size // n_labels
    bv = behavior_vector.reshape(n_tests, n_labels)

    # 获取每个测试用例的预测标签索引
    predicted_indices = np.argmax(bv, axis=1)
    
    # 生成每个测试用例的Oracle预期结果
    expected_outputs = []
    for a, b, c in lhs_tests[:n_tests]:
        expected = oracle(a, b, c)
        expected_outputs.append(label2idx[expected])
    expected_outputs = np.array(expected_outputs)
    
    # 如果存在任何预测与Oracle结果不一致，则变异体被杀死
    all_correct = np.all(predicted_indices == expected_outputs)
    return not all_correct


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


def compute_reduced_mutation_score(coverage_vectors: Dict[str, np.ndarray],
                                   dict_represent: Dict[str, str]):
    """
    基于约简代表变异体的 Mutation Score
    dict_represent: {cluster_0: M01, cluster_1: M21, ...}
    """
    killed = 0
    valid_clusters = 0  # 只统计有覆盖向量的簇

    for cluster, mid in dict_represent.items():
        if mid not in coverage_vectors:
            continue  # 跳过没有行为向量的簇
        valid_clusters += 1
        if is_mutant_killed(coverage_vectors[mid]):
            killed += 1

    score = killed / valid_clusters if valid_clusters > 0 else 0.0

    return {
        "clusters": valid_clusters,
        "killed_representatives": killed,
        "reduced_mutation_score": score
    }


def compute_test_averaged_ms(coverage_vectors: Dict[str, np.ndarray]):
    """
    每个 test 能杀死多少比例的变异体
    """
    n_labels = len(LABELS)

    # 从任意一个 mutant 推断 n_tests
    any_vec = next(iter(coverage_vectors.values()))

    if any_vec.size % n_labels != 0:
        raise ValueError("Behavior vector size not divisible by label count")

    n_tests = any_vec.size // n_labels

    # 生成每个测试用例的Oracle预期结果
    expected_outputs = []
    for a, b, c in lhs_tests[:n_tests]:
        expected = oracle(a, b, c)
        expected_outputs.append(label2idx[expected])
    expected_outputs = np.array(expected_outputs)

    killed_per_test = np.zeros(n_tests)

    for vec in coverage_vectors.values():
        if vec.size != any_vec.size:
            raise ValueError("Inconsistent behavior vector sizes across mutants")

        bv = vec.reshape(n_tests, n_labels)
        predicted_indices = np.argmax(bv, axis=1)
        
        # 如果预测与Oracle结果不一致，则该测试杀死了这个变异体
        killed_per_test += (predicted_indices != expected_outputs).astype(int)

    killed_ratio = killed_per_test / len(coverage_vectors)

    return {
        "n_tests": n_tests,
        "mean_test_ms": float(np.mean(killed_ratio)),
        "std_test_ms": float(np.std(killed_ratio)),
        "min_test_ms": float(np.min(killed_ratio)),
        "max_test_ms": float(np.max(killed_ratio)),
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
    