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

LABELS = ["Invalid", "Equilateral", "Isosceles", "Scalene"]
label2idx = {l: i for i, l in enumerate(LABELS)}

def oracle(a: int, b: int, c: int) -> str:
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"

def generate_stratified_triangle_tests(n=10):
    """分层生成测试用例，确保四类三角形均匀分布"""
    n_per_class = n // 4
    
    tests = []
    
    # 1. Invalid类
    sampler = qmc.LatinHypercube(d=3)
    invalid_samples = sampler.random(n_per_class)
    for s in invalid_samples:
        c = np.random.randint(10, 50)
        a = int(s[0] * (c-1)) + 1
        b = int(s[1] * (c-a)) + 1
        if a + b >= c:
            b = max(1, c - a - 1)
        tests.append([a, b, c])
    
    # 2. Equilateral类
    for _ in range(n_per_class):
        side = np.random.randint(10, 50)
        tests.append([side, side, side])
    
    # 3. Isosceles类
    sampler = qmc.LatinHypercube(d=2)
    iso_samples = sampler.random(n_per_class)
    for s in iso_samples:
        equal_side = int(s[0] * 30) + 10
        base = int(s[1] * (2*equal_side - 1)) + 1
        if base == equal_side:
            base = (base + 1) % (2*equal_side) + 1
        tests.append([equal_side, equal_side, base])
    
    # 4. Scalene类
    sampler = qmc.LatinHypercube(d=3)
    sca_samples = sampler.random(n_per_class)
    for s in sca_samples:
        a = int(s[0] * 20) + 10
        b = int(s[1] * 20) + 10
        c_min = abs(a-b) + 1
        c_max = a + b - 1
        c = int(s[2] * (c_max - c_min)) + c_min
        if c == a or c == b:
            c = c_min + (c_max - c_min)//2
        tests.append([a, b, c])
    
    np.random.shuffle(tests)
    return np.array(tests)

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
            if out is None:
                out = "Invalid"
        except Exception:
            out = "Invalid"
        v = np.zeros(len(LABELS))
        if out in label2idx:
            v[label2idx[out]] = 1
        else:
            v[label2idx["Invalid"]] = 1
        cov.append(v)
    return np.concatenate(cov)

def generate_coverage_vectors(lhs_tests: np.ndarray):
    coverage_vectors = {}
    for mid in MUTANT_OPERATOR_MAP:
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            continue
        triangle_func = load_triangle_func(py_file)
        coverage_vectors[mid] = run_tests(triangle_func, lhs_tests)
    return coverage_vectors

def is_mutant_killed(behavior_vector: np.ndarray, lhs_tests: np.ndarray, oracle_func):
    n_tests = len(lhs_tests)
    bv = behavior_vector.reshape(n_tests, len(LABELS))
    predicted_indices = np.argmax(bv, axis=1)
    expected_indices = np.array([label2idx[oracle_func(a, b, c)] for a, b, c in lhs_tests])
    return not np.all(predicted_indices == expected_indices)

def extract_kill_matrix_and_profiles(coverage_vectors, lhs_tests, oracle_func):
    mutant_ids = sorted(coverage_vectors.keys())
    n_mutants = len(mutant_ids)
    n_tests = len(lhs_tests)
    
    kill_matrix = np.zeros((n_mutants, n_tests), dtype=int)
    behavior_profiles = np.zeros((n_mutants, len(LABELS)))
    expected_indices = np.array([label2idx[oracle_func(a, b, c)] for a, b, c in lhs_tests])
    
    for i, mid in enumerate(mutant_ids):
        vec = coverage_vectors[mid]
        bv = vec.reshape(n_tests, len(LABELS))
        predicted_indices = np.argmax(bv, axis=1)
        is_killed_per_test = (predicted_indices != expected_indices).astype(int)
        kill_matrix[i] = is_killed_per_test
        behavior_profiles[i] = bv.mean(axis=0)
    
    return kill_matrix, behavior_profiles, mutant_ids

def five_minute_diagnosis(kill_matrix, behavior_profiles):
    n_mutants, n_tests = kill_matrix.shape
    print(f"\n{'='*60}")
    print(f"【5分钟诊断报告】样本: {n_mutants}变异体 × {n_tests}测试用例")
    print(f"{'='*60}")
    
    # 诊断A
    sample_size = min(50, n_mutants)
    indices = np.random.choice(n_mutants, sample_size, replace=False)
    jaccard_sims = []
    for i, j in combinations(indices, 2):
        intersection = np.sum((kill_matrix[i] == 1) & (kill_matrix[j] == 1))
        union = np.sum((kill_matrix[i] == 1) | (kill_matrix[j] == 1))
        if union > 0:
            jaccard_sims.append(intersection / union)
    avg_jaccard = np.mean(jaccard_sims) if jaccard_sims else 0
    print(f"\n【诊断A】平均Jaccard: {avg_jaccard:.3f}")
    risk_a = "HIGH" if avg_jaccard < 0.15 else ("MEDIUM" if avg_jaccard < 0.3 else "LOW")
    print(f"  风险: {risk_a}")

    # 诊断B
    kill_labels = (kill_matrix.sum(axis=1) > 0).astype(int)
    dominant_errors = behavior_profiles.argmax(axis=1)
    mi = mutual_info_score(dominant_errors, kill_labels)
    print(f"\n【诊断B】互信息: {mi:.3f}")
    risk_b = "HIGH" if mi > 0.7 else ("MEDIUM" if mi > 0.4 else "LOW")
    print(f"  风险: {risk_b}")

    # 诊断C
    error_counts = np.bincount(dominant_errors, minlength=len(LABELS))
    unique_errors = np.sum(error_counts > 0)
    probs = error_counts / np.sum(error_counts)
    error_entropy = entropy(probs, base=2) / np.log2(len(LABELS))
    print(f"\n【诊断C】出现类别: {unique_errors}/{len(LABELS)}, 熵: {error_entropy:.3f}")
    print(f"  分布: {dict(zip(LABELS, error_counts))}")
    risk_c = "HIGH" if unique_errors <= 2 or error_entropy < 0.3 else ("MEDIUM" if unique_errors <= 3 else "LOW")
    print(f"  风险: {risk_c}")

    risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    total_score = risk_map[risk_a] + risk_map[risk_b] + risk_map[risk_c]
    decision = "STOP" if total_score >= 5 else ("CAUTION" if total_score >= 3 else "GO")
    print(f"\n【决策】{total_score}/6 ({decision})")
    return decision == "GO"

def cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, force_one_per_class=False):
    """
    force_one_per_class=True: 每类强制只选1个（用于k=4对比）
    """
    dominant_errors = behavior_profiles.argmax(axis=1)
    representatives = {}
    cluster_id = 0
    
    error_groups = defaultdict(list)
    for i, mid in enumerate(mutant_ids):
        error_type = dominant_errors[i]
        error_groups[error_type].append((i, mid))
    
    print(f"\n行为预分组：{len(error_groups)}个主导类别组")
    
    for error_type, members in error_groups.items():
        if len(members) == 1:
            idx, mid = members[0]
            representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [mid]
            cluster_id += 1
        else:
            indices = [m[0] for m in members]
            mids = [m[1] for m in members]
            sub_kill_matrix = kill_matrix[indices]
            
            if force_one_per_class or len(members) <= 3:
                # 强制选1个：选杀死最多的
                kill_counts = sub_kill_matrix.sum(axis=1)
                best_idx = np.argmax(kill_counts)
                representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [mids[best_idx]]
                cluster_id += 1
            else:
                # 分为2个子簇
                sub_k = 2
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

def baseline_cluster(kill_matrix, mutant_ids, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(kill_matrix)
    representatives = {}
    for i in range(k):
        mask = (labels == i)
        cluster_indices = np.where(mask)[0]
        if len(cluster_indices) > 0:
            kill_counts = kill_matrix[cluster_indices].sum(axis=1)
            best_idx = cluster_indices[np.argmax(kill_counts)]
            representatives[f"cluster_{i}"] = [mutant_ids[best_idx]]
    return representatives

def evaluate_reduction(coverage_vectors, representatives, mutant_ids, lhs_tests, oracle_func):
    """
    修正后的函数：显式接收所有必需参数
    """
    n_original = len(mutant_ids)
    n_reduced = len(representatives)
    n_tests = len(lhs_tests)
    expected_indices = np.array([label2idx[oracle_func(a, b, c)] for a, b, c in lhs_tests])
    
    def get_fault_type_indices(mid):
        if mid not in coverage_vectors:
            return set()
        vec = coverage_vectors[mid].reshape(n_tests, len(LABELS))
        predicted = np.argmax(vec, axis=1)
        fault_mask = (predicted != expected_indices)
        if not np.any(fault_mask):
            return set()
        return set(predicted[fault_mask])
    
    all_fault_types = set()
    for mid in mutant_ids:
        all_fault_types.update(get_fault_type_indices(mid))
    
    rep_fault_types = set()
    for cluster, mids in representatives.items():
        for mid in mids:
            rep_fault_types.update(get_fault_type_indices(mid))
    
    ftrr = len(rep_fault_types) / len(all_fault_types) if all_fault_types else 1.0
    
    # 计算MS
    killed_full = sum(1 for mid in mutant_ids if is_mutant_killed(coverage_vectors[mid], lhs_tests, oracle_func))
    original_ms = killed_full / len(mutant_ids)
    
    killed_reduced = sum(1 for cluster, mids in representatives.items() 
                        for mid in mids if is_mutant_killed(coverage_vectors[mid], lhs_tests, oracle_func))
    reduced_ms = killed_reduced / len(representatives) if representatives else 0
    
    return {
        'original_ms': original_ms,
        'reduced_ms': reduced_ms,
        'reduction_ratio': 1 - n_reduced/n_original,
        'ftrr': ftrr,
        'all_fault_types_count': len(all_fault_types),
        'rep_fault_types_count': len(rep_fault_types),
        'representatives': representatives
    }

def main():
    N_TESTS = 20
    print("步骤1: 生成分层LHS测试用例...")
    lhs_tests = generate_stratified_triangle_tests(n=N_TESTS)
    
    print("步骤2: 生成覆盖向量...")
    coverage_vectors = generate_coverage_vectors(lhs_tests)
    
    print("步骤3: 解析数据...")
    kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(
        coverage_vectors, lhs_tests, oracle
    )
    
    print("步骤4: 诊断...")
    is_go = five_minute_diagnosis(kill_matrix, behavior_profiles)
    
    # ========== 关键修正：k=4对比实验 ==========
    print(f"\n{'='*60}")
    print("【方案A：极端约简对比 (k=4)】")
    print(f"{'='*60}")
    
    # 对照组：纯K-Means，k=4
    print("\n对照组 (K-Means, k=4):")
    reps_base_k4 = baseline_cluster(kill_matrix, mutant_ids, k=4)
    eval_base_k4 = evaluate_reduction(
        coverage_vectors, 
        reps_base_k4, 
        mutant_ids, 
        lhs_tests, 
        oracle
    )
    print(f"  选出代表数: {len(reps_base_k4)}")
    print(f"  FTRR: {eval_base_k4['ftrr']:.1%} ({eval_base_k4['rep_fault_types_count']}/{eval_base_k4['all_fault_types_count']}类)")
    print(f"  代表: {list(reps_base_k4.values())}")
    
    # 实验组：行为约束，强制每类1个（共4个）
    print("\n实验组 (行为约束, 每类1个):")
    reps_exp_k4 = cluster_with_behavior_constraint(
        kill_matrix, behavior_profiles, mutant_ids, force_one_per_class=True
    )
    eval_exp_k4 = evaluate_reduction(
        coverage_vectors, 
        reps_exp_k4, 
        mutant_ids, 
        lhs_tests, 
        oracle
    )
    print(f"  选出代表数: {len(reps_exp_k4)}")
    print(f"  FTRR: {eval_exp_k4['ftrr']:.1%} ({eval_exp_k4['rep_fault_types_count']}/{eval_exp_k4['all_fault_types_count']}类)")
    print(f"  代表: {list(reps_exp_k4.values())}")
    
    # 对比结果
    print(f"\n【k=4对比结果】")
    print(f"  对照组 FTRR: {eval_base_k4['ftrr']:.1%}")
    print(f"  实验组 FTRR: {eval_exp_k4['ftrr']:.1%}")
    if eval_exp_k4['ftrr'] > eval_base_k4['ftrr']:
        print(f"  ✅ 行为指纹方法在极端约简下更优 (提升 {eval_exp_k4['ftrr']-eval_base_k4['ftrr']:.1%})")
    elif eval_exp_k4['ftrr'] == eval_base_k4['ftrr']:
        print(f"  ⚠️ 两者等价 (测试集可能过强)")
    else:
        print(f"  ❌ 异常：对照组更优")



# 很重要的一点，当测试用例不足的时候，行文指纹可以起到很好的作用。
if __name__ == "__main__":
    main()