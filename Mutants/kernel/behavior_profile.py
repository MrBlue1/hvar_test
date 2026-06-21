import os
import sys
import importlib.util
import numpy as np
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from itertools import combinations
from scipy.stats import entropy, qmc
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

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
# 行为标签（RBF Kernel专用，9类细粒度行为）
# =====================================================
# =====================================================
# 【优化】行为标签（10维语义明确的正交分类）
# =====================================================
LABELS = [
    "CORRECT",           # 0: 完全正确
    "NAN_PROPAGATION",   # 1: NaN污染（0*Inf或无效运算）
    "POSITIVE_OVERFLOW", # 2: 数值上溢（exp过大）
    "UNDERFLOW",         # 3: 数值下溢（exp过小视为0，可接受但需记录）
    "NEGATIVE_VALUE",    # 4: 负值错误（RBF应非负）
    "ABOVE_ONE",         # 5: 超1.0错误（RBF最大值应为1）
    "ASYMMETRY",         # 6: 非对称错误（K≠K^T）
    "DIAGONAL_ERROR",    # 7: 对角线≠1（当X==Y时）
    "SHAPE_MISMATCH",    # 8: 输出维度错误
    "TYPE_ERROR"         # 9: 返回类型错误
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
# LHS 测试用例生成
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
# 【优化】行为分类（RBF Kernel专用 - 严格优先级）
# =====================================================
def classify_behavior(expected, actual, X, Y, exception=None) -> str:
    """
    RBF核函数行为分类器（优先级：异常>类型>形状>数值>数学性质）
    返回: LABELS中的标签字符串
    """
    n_x = X.shape[0]
    n_y = Y.shape[0] if Y is not None else n_x
    is_symmetric_case = (Y is None) or (Y is X) or np.array_equal(X, Y)
    
    # 1. 异常捕获（最高优先级）
    if exception is not None:
        if isinstance(exception, TypeError):
            return "TYPE_ERROR"
        elif isinstance(exception, (ValueError, IndexError, AttributeError)):
            return "SHAPE_MISMATCH"
        else:
            # 捕获溢出警告等被转换为异常的情况
            return "POSITIVE_OVERFLOW" if "overflow" in str(exception).lower() else "NAN_PROPAGATION"
    
    # 2. 类型检查
    if not isinstance(actual, np.ndarray):
        return "TYPE_ERROR"
    
    # 3. 形状检查（必须在数值检查前，否则后续操作可能崩溃）
    if actual.shape != (n_x, n_y):
        return "SHAPE_MISMATCH"
    
    # 4. 数值异常细分（关键优化：区分Inf和NaN）
    has_inf = np.any(np.isinf(actual))
    has_nan = np.any(np.isnan(actual))
    
    if has_nan:
        return "NAN_PROPAGATION"  # 通常由 0 * Inf 或 Inf/Inf 产生
    if has_inf:
        return "POSITIVE_OVERFLOW"  # exp(gamma * large_number) 上溢
    
    # 5. 范围检查（RBF值域 [0,1]）
    min_val, max_val = actual.min(), actual.max()
    
    if min_val < -EPS:
        return "NEGATIVE_VALUE"  # 严重错误：RBF不可能为负
    elif min_val < 0:  # 在-EPS和0之间
        return "UNDERFLOW"  # 微小的数值下溢，接近0的负数
    
    if max_val > 1.0 + VALUE_TOL:
        return "ABOVE_ONE"  # 超过理论最大值1
    
    # 6. 数学性质检查（仅当X==Y时严格检查对角线）
    if is_symmetric_case:
        # 检查对称性
        if not np.allclose(actual, actual.T, atol=SYMMETRY_TOL):
            return "ASYMMETRY"
        
        # 检查对角线（K(x,x) 应严格等于 1）
        diag = np.diag(actual)
        if not np.allclose(diag, 1.0, atol=DIAGONAL_TOL):
            return "DIAGONAL_ERROR"
    
    # 7. 通过所有检查
    return "CORRECT"


# =====================================================
# 【新增】辅助检查函数（可选，用于增强数值稳定性检测）
# =====================================================
def check_numerical_stability(X, Y, gamma, actual):
    """
    检测数值稳定性问题（如下溢视为0，这在RBF中是可接受的，但需记录）
    """
    if not isinstance(actual, np.ndarray):
        return None
    
    # 检测是否所有值都接近0（可能 gamma 过小或下溢）
    if actual.max() < 1e-15 and actual.min() >= 0:
        return "UNDERFLOW"
    return None

# =====================================================
# 【修改】向量生成（适配新LABELS和分类器）
# =====================================================
def generate_coverage_vectors(n_tests: int, mutants_dir: Optional[Path] = None, seed: int = 42):
    """生成覆盖向量（维度 = len(LABELS) = 10）"""
    if mutants_dir is None:
        mutants_dir = Path(__file__).parent
    tests = lhs_samples(n_tests, seed=seed)
    oracle_kernel = load_kernel_func(mutants_dir / f"{ORACLE_ID}.py")
    
    coverage_data = {}
    n_labels = len(LABELS)  # 现在是 10
    
    # 预建映射加速
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    
    for mid in MUTANT_OPERATOR_MAP:
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            continue

        kernel_func = load_kernel_func(py_file)
        ms_results = []
        behavior_results = []
        
        for X, Y, gamma, eps in tests:
            exception = None
            actual = None
            
            try:
                with np.errstate(all="ignore"):  # 我们手动处理数值异常
                    actual = kernel_func(X, Y, gamma, eps)
            except Exception as e:
                exception = e
            
            # 行为分类（使用优化后的分类器）
            label = classify_behavior(None, actual, X, Y, exception)
            v_detail = np.zeros(n_labels)
            v_detail[label_to_idx[label]] = 1.0  # One-hot编码
            behavior_results.append(v_detail)
            
            # MS判断（保持原有逻辑）
            is_killed = False
            if exception is not None:
                is_killed = True
            elif not isinstance(actual, np.ndarray):
                is_killed = True
            elif actual.shape != (X.shape[0], (Y.shape[0] if Y is not None else X.shape[0])):
                is_killed = True
            else:
                try:
                    with np.errstate(all="ignore"):
                        ref = oracle_kernel(X, Y, gamma, eps)
                        if not np.allclose(actual, ref, atol=1e-6, rtol=1e-5):
                            is_killed = True
                except:
                    is_killed = True
            
            v_ms = np.zeros(2)
            v_ms[1 if is_killed else 0] = 1
            ms_results.append(v_ms)
        
        coverage_data[mid] = {
            'ms': np.concatenate(ms_results),
            'behavior': np.concatenate(behavior_results)  # 现在维度是 n_tests * 10
        }

    return coverage_data


# =====================================================
# 【新增】数据解析：提取杀死矩阵和行为指纹
# =====================================================
def extract_kill_matrix_and_profiles(coverage_data: Dict[str, Dict]):
    """提取杀死矩阵和行为指纹（自动适配维度）"""
    mutant_ids = sorted(coverage_data.keys())
    n_mutants = len(mutant_ids)
    n_labels = len(LABELS)  # 动态获取维度
    
    first_ms = coverage_data[mutant_ids[0]]['ms']
    n_tests = first_ms.size // 2
    
    kill_matrix = np.zeros((n_mutants, n_tests), dtype=int)
    behavior_profiles = np.zeros((n_mutants, n_labels))  # 使用 n_labels
    
    for i, mid in enumerate(mutant_ids):
        data = coverage_data[mid]
        ms_vec = data['ms'].reshape(n_tests, 2)
        kill_matrix[i] = ms_vec[:, 1]
        
        beh_vec = data['behavior'].reshape(n_tests, n_labels)  # 使用 n_labels
        behavior_profiles[i] = beh_vec.mean(axis=0)
    
    return kill_matrix, behavior_profiles, mutant_ids


def five_minute_diagnosis(kill_matrix: np.ndarray, behavior_profiles: np.ndarray) -> Dict:
    """5分钟诊断（自动适配 LABELS 维度）"""
    n_mutants, n_tests = kill_matrix.shape
    n_labels = len(LABELS)  # 自动适配，不再是硬编码7或9
    
    print(f"\n{'='*60}")
    print(f"【5分钟诊断报告】样本: {n_mutants}变异体 × {n_tests}测试用例")
    print(f"行为类别数: {n_labels}")
    print(f"{'='*60}")
    
    # 诊断A：测试集完备性（保持不变）
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
        risk_a, msg_a = "HIGH", "测试集过强，指纹增益有限"
    elif avg_jaccard < 0.3:
        risk_a, msg_a = "MEDIUM", "区分度较好"
    else:
        risk_a, msg_a = "LOW", "存在混淆，需要指纹辅助"
    print(f"  平均Jaccard: {avg_jaccard:.3f} - {risk_a}: {msg_a}")

    # 诊断B：模态相关性（保持不变）
    print("\n【诊断B】行为指纹与杀死矩阵相关性")
    kill_labels = (kill_matrix.sum(axis=1) > 0).astype(int)
    dominant_errors = behavior_profiles.argmax(axis=1)
    mi = mutual_info_score(dominant_errors, kill_labels)
    profile_variance = np.mean(np.var(behavior_profiles, axis=0))
    
    print(f"  互信息: {mi:.3f}, 方差: {profile_variance:.3f}")
    
    if mi > 0.7 or profile_variance < 0.01:
        risk_b = "HIGH"
    elif mi > 0.4:
        risk_b = "MEDIUM"
    else:
        risk_b = "LOW"
    print(f"  风险: {risk_b}")

    # 诊断C：行为多样性（自动适配 n_labels）
    print("\n【诊断C】细粒度行为多样性")
    error_counts = np.bincount(dominant_errors, minlength=n_labels)
    unique_errors = np.sum(error_counts > 0)
    probs = error_counts / np.sum(error_counts)
    error_entropy = entropy(probs, base=2) / np.log2(n_labels) if n_labels > 1 else 0
    
    print(f"  出现类别: {unique_errors}/{n_labels}, 熵: {error_entropy:.3f}")
    print(f"  分布: {dict(zip(LABELS, error_counts))}")
    
    if unique_errors <= 3 or error_entropy < 0.4:
        risk_c = "HIGH"
    elif unique_errors <= 5:
        risk_c = "MEDIUM"
    else:
        risk_c = "LOW"
    print(f"  风险: {risk_c}")

    # 综合决策（保持不变）
    risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    total_score = risk_map[risk_a] + risk_map[risk_b] + risk_map[risk_c]
    
    print(f"\n{'='*60}")
    if total_score >= 5:
        decision = "STOP"
        print(f"【决策】{total_score}/6 ({decision}) - 建议转向其他方法")
    elif total_score >= 3:
        decision = "CAUTION"
        print(f"【决策】{total_score}/6 ({decision}) - 谨慎进行")
    else:
        decision = "GO"
        print(f"【决策】{total_score}/6 ({decision}) - 全速前进")
    print(f"{'='*60}")
    
    return {
        'risk_a': risk_a, 'risk_b': risk_b, 'risk_c': risk_c,
        'total_score': total_score, 'decision': decision,
        'metrics': {'avg_jaccard': avg_jaccard, 'mi': mi, 'error_entropy': error_entropy}
    }
# ============================================
# 7. 【新增】聚类与评估模块
# ============================================
def cluster_with_behavior_constraint(kill_matrix: np.ndarray, 
                                    behavior_profiles: np.ndarray,
                                    mutant_ids: List[str],
                                    k: int,
                                    force_one_per_class: bool = False) -> Dict[str, List[str]]:
    """
    基于细粒度行为的约束聚类（严格遵守 k 限制）
    """
    dominant_errors = behavior_profiles.argmax(axis=1)
    n_mutants = len(mutant_ids)
    
    representatives = {}
    cluster_id = 0
    
    # 按细粒度行为分组（7类）
    error_groups = defaultdict(list)
    for i, mid in enumerate(mutant_ids):
        error_type = dominant_errors[i]
        error_groups[error_type].append((i, mid))
    
    print(f"\n行为预分组：{len(error_groups)}个细粒度类别组")
    for etype, members in sorted(error_groups.items()):
        print(f"  {LABELS[etype]}: {len(members)}个变异体")
    
    # 【关键修改】如果类别数 > k，优先保留错误类（非"正确"类）
    if len(error_groups) > k:
        print(f"\n⚠️ 类别数({len(error_groups)}) > k({k})，启用优先级策略")
        
        # 将"正确"类（索引0）设为低优先级，其他错误类高优先级
        prioritized_groups = []
        for error_type, members in error_groups.items():
            priority = 1 if error_type == 0 else 0  # 0=高优先级, 1=低优先级
            prioritized_groups.append((priority, len(members), error_type, members))
        
        # 排序：先按优先级（错误类在前），再按数量（小众优先）
        prioritized_groups.sort(key=lambda x: (x[0], x[1]))
        
        # 只取前 k 个组
        selected_groups = [(et, members) for _, _, et, members in prioritized_groups[:k]]
        print(f"  选中 {len(selected_groups)} 个高优先级组（已丢弃{len(error_groups)-k}个低优先级组）")
    else:
        selected_groups = list(error_groups.items())
    
    # 从选中的组中提取代表
    for error_type, members in selected_groups:
        if len(members) == 1:
            idx, mid = members[0]
            representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [mid]
            cluster_id += 1
        else:
            indices = [m[0] for m in members]
            mids = [m[1] for m in members]
            sub_kill_matrix = kill_matrix[indices]
            
            if force_one_per_class or len(members) <= 3:
                # 选杀死最多的
                kill_counts = sub_kill_matrix.sum(axis=1)
                best_idx = np.argmax(kill_counts)
                representatives[f"C{cluster_id}_{LABELS[error_type]}"] = [mids[best_idx]]
                cluster_id += 1
            else:
                # 成员较多，但在组内聚类为 min(2, len(members)//2) 个
                sub_k = min(2, len(members) // 2)
                kmeans = KMeans(n_clusters=sub_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sub_kill_matrix)
                
                for sub_id in range(sub_k):
                    mask = (labels == sub_id)
                    if np.any(mask):
                        cluster_mids = [mids[j] for j in range(len(mids)) if mask[j]]
                        sub_kills = sub_kill_matrix[mask].sum(axis=1)
                        best_idx = np.argmax(sub_kills)
                        representatives[f"C{cluster_id}_{LABELS[error_type]}_sub{sub_id}"] = [cluster_mids[best_idx]]
                        cluster_id += 1
    
    # 【安全检查】如果代表数超过 k，强制截断（按杀死数排序保留前k个）
    if len(representatives) > k:
        print(f"\n⚠️ 代表数({len(representatives)})超过k({k})，强制截断...")
        # 计算每个代表簇的杀死数
        kill_counts_map = {}
        for cid, mids in representatives.items():
            mid = mids[0]
            if mid in mutant_ids:
                idx = mutant_ids.index(mid)
                kill_counts_map[cid] = kill_matrix[idx].sum()
        
        # 按杀死数排序，保留前 k 个
        sorted_reps = sorted(kill_counts_map.items(), key=lambda x: -x[1])[:k]
        filtered_reps = {cid: representatives[cid] for cid, _ in sorted_reps}
        representatives = filtered_reps
    

        # ============================================
    # 【新增】预算填充：利用剩余名额提升 MS 保持率
    # ============================================
    remaining_slots = k - len(representatives)
    
    if remaining_slots > 0:
        print(f"\n【预算填充】剩余 {remaining_slots} 个名额，从大类中补充高杀死率代表...")
        
        # 获取当前已选中的代表 ID 集合，避免重复
        selected_mids = set()
        for mids in representatives.values():
            selected_mids.update(mids)
        
        # 从 error_groups 中找出还有剩余变异体的组（排除已选中的）
        candidates = []
        for error_type, members in error_groups.items():
            # 找出该组中未被选中的变异体
            unselected = [(idx, mid) for idx, mid in members if mid not in selected_mids]
            if unselected:
                candidates.append((error_type, unselected))
        
        # 按组大小排序（大组优先，有更多选择空间）
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        
        slots_filled = 0
        for error_type, unselected_members in candidates:
            if slots_filled >= remaining_slots:
                break
            
            # 在这些候选者中选 kill count 最高的
            indices = [m[0] for m in unselected_members]
            mids = [m[1] for m in unselected_members]
            sub_kill = kill_matrix[indices]
            
            kill_counts = sub_kill.sum(axis=1)
            best_local_idx = np.argmax(kill_counts)
            best_mid = mids[best_local_idx]
            best_kill_count = kill_counts[best_local_idx]
            
            # 添加为代表
            cluster_name = f"C{cluster_id}_{LABELS[error_type]}_sup"
            representatives[cluster_name] = [best_mid]
            cluster_id += 1
            slots_filled += 1
            
            print(f"  + {best_mid} ({LABELS[error_type]}, 杀死{best_kill_count}个测试用例)")
        
        print(f"  实际填充: {slots_filled} 个，最终代表数: {len(representatives)}")

    print(f"\n【实验组约束】最终选中 {len(representatives)} 个代表（预算k={k}）")
    return representatives

    print(f"\n【实验组约束】最终选中 {len(representatives)} 个代表（预算k={k}）")
    return representatives


def baseline_cluster(kill_matrix, mutant_ids, k):
    """纯杀死矩阵K-Means"""
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

def evaluate_reduction(coverage_data, representatives, mutant_ids):
    """计算约简质量（维度自动适配）"""
    n_original = len(mutant_ids)
    n_reduced = len(representatives)
    n_labels = len(LABELS)  # 动态获取维度
    
    def get_dominant_behavior(mid):
        if mid not in coverage_data:
            return None
        beh_vec = coverage_data[mid]['behavior'].reshape(-1, n_labels)  # 使用 n_labels
        return np.argmax(beh_vec.mean(axis=0))
    
    # FTRR 计算（保持不变）
    all_behavior_types = set()
    for mid in mutant_ids:
        dom = get_dominant_behavior(mid)
        if dom is not None:
            all_behavior_types.add(dom)
    
    rep_behavior_types = set()
    for cluster, mids in representatives.items():
        for mid in mids:
            dom = get_dominant_behavior(mid)
            if dom is not None:
                rep_behavior_types.add(dom)
    
    ftrr = len(rep_behavior_types) / len(all_behavior_types) if all_behavior_types else 1.0
    
    # MS 计算逻辑（保持不变）
    killed_original_set = set()
    for mid in mutant_ids:
        if np.any(coverage_data[mid]['ms'].reshape(-1, 2)[:, 1] == 1):
            killed_original_set.add(mid)
    n_killed_original = len(killed_original_set)
    original_ms = n_killed_original / len(mutant_ids)
    
    rep_set = set()
    for mids in representatives.values():
        rep_set.update(mids)
    
    killed_retained_set = killed_original_set & rep_set
    n_killed_retained = len(killed_retained_set)
    
    ms_retention = n_killed_retained / n_killed_original if n_killed_original else 0
    
    killed_in_rep = sum(1 for m in rep_set if m in killed_original_set)
    rep_purity = killed_in_rep / len(rep_set) if rep_set else 0
    
    return {
        'original_ms': original_ms,
        'reduced_ms': ms_retention,
        'rep_purity': rep_purity,
        'n_killed_original': n_killed_original,
        'n_killed_retained': n_killed_retained,
        'reduction_ratio': 1 - n_reduced/n_original,
        'ftrr': ftrr,
        'all_behavior_types': [LABELS[i] for i in all_behavior_types],
        'rep_behavior_types': [LABELS[i] for i in rep_behavior_types],
        'representatives': representatives,
        'killed_retained_ids': list(killed_retained_set)
    }


def auto_determine_k(behavior_profiles: np.ndarray, 
                    mutant_ids: List[str],
                    min_per_class: int = 1,
                    budget_ratio: float = 0.15,
                    max_ratio: float = 0.5) -> int:
    """
    自动确定聚类数量 k：
    - 保证每类至少 min_per_class 个代表
    - 但总不超过总变异体数的 budget_ratio 比例（默认15%）
    - 且不超过 max_ratio（默认50%）
    """
    dominant_errors = behavior_profiles.argmax(axis=1)
    unique_behaviors = len(set(dominant_errors))
    n_mutants = len(mutant_ids)
    
    # 基础预算：每类至少 min_per_class 个
    base_k = unique_behaviors * min_per_class
    
    # 比例预算：基于总变异体数
    ratio_k = int(n_mutants * budget_ratio)
    
    # 最终 k：取基础预算和比例预算的较大值，但不超过上限
    k = min(max(base_k, ratio_k), int(n_mutants * max_ratio))
    
    # 确保至少为 1
    k = max(k, 1)
    
    print(f"\n【自动预算计算】")
    print(f"  总变异体: {n_mutants}")
    print(f"  行为类别: {unique_behaviors}")
    print(f"  基础预算: {base_k} (每类{min_per_class}个)")
    print(f"  比例预算: {ratio_k} ({budget_ratio*100:.0f}%)")
    print(f"  最终 k值: {k}")
    
    return k

# ============================================
# 8. 主流程（Kernel验证专用）
# ============================================
def main():
    # 配置
    N_TESTS = 200  # 从20开始测试（资源受限场景）
    SEED = 42
    
    print("=" * 60)
    print("Kernel 行为指纹验证实验")
    print("=" * 60)
    
    # 步骤1: 生成数据（同时包含MS向量和行为指纹）
    print(f"\n步骤1: 生成 {N_TESTS} 个测试用例的行为数据...")
    coverage_data = generate_coverage_vectors(n_tests=N_TESTS, seed=SEED)
    
    # 步骤2: 解析数据
    print("\n步骤2: 解析杀死矩阵与行为指纹...")
    kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(coverage_data)
    
    # 步骤3: 诊断
    print("\n步骤3: 运行5分钟诊断...")
    diagnosis = five_minute_diagnosis(kill_matrix, behavior_profiles)
    
        # 步骤4: 自动k值对比实验
    print(f"\n{'='*60}")
    print("【自动k值极端约简对比】")
    print(f"{'='*60}")
    
    # 【关键】自动确定 k（例如保留15%的变异体，或至少每类1个）
    k_auto = auto_determine_k(behavior_profiles, mutant_ids, 
                              min_per_class=1, budget_ratio=0.15)
    
    # 对照组（使用相同的k_auto）
    print(f"\n对照组 (纯Kill Matrix K-Means, k={k_auto}):")
    reps_base = baseline_cluster(kill_matrix, mutant_ids, k=k_auto)
    eval_base = evaluate_reduction(coverage_data, reps_base, mutant_ids)
    print(f"  代表数: {len(reps_base)}")
    print(f"  FTRR: {eval_base['ftrr']:.1%} ({len(eval_base['rep_behavior_types'])}/{len(eval_base['all_behavior_types'])}类)")
    print(f"  保留类别: {eval_base['rep_behavior_types']}")
    
    # 实验组（现在传入k_auto并严格遵守）
    print(f"\n实验组 (行为约束聚类, k={k_auto}):")
    reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, 
                                                k=k_auto, force_one_per_class=True)
    eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
    print(f"  代表数: {len(reps_exp)}")
    print(f"  FTRR: {eval_exp['ftrr']:.1%} ({len(eval_exp['rep_behavior_types'])}/{len(eval_exp['all_behavior_types'])}类)")
    print(f"  保留类别: {eval_exp['rep_behavior_types']}")
    
    # 对比结果
    print(f"\n{'='*60}")
    print("【对比结果】")
    print(f"预算k值: {k_auto}")
    print(f"{'='*60}")
    
    # FTRR 对比
    print(f"\n【故障类型覆盖率 (FTRR)】")
    print(f"对照组: {eval_base['ftrr']:.1%} ({len(eval_base['rep_behavior_types'])}/{len(eval_base['all_behavior_types'])}类)")
    print(f"实验组: {eval_exp['ftrr']:.1%} ({len(eval_exp['rep_behavior_types'])}/{len(eval_exp['all_behavior_types'])}类)")
    
    # 【新增】MS 保持率对比
    # 【修正后】MS 保持率对比
    print(f"\n【变异分数保持率 (MS Retention)】")
    print(f"原始全集: {eval_base['original_ms']:.1%} ({eval_base['n_killed_original']}/60个被杀)")
    
    print(f"\n对照组:")
    print(f"  保留杀死数: {eval_base['n_killed_retained']}/{eval_base['n_killed_original']}")
    print(f"  MS 保持率: {eval_base['reduced_ms']:.1%}")
    print(f"  代表纯度: {eval_base['rep_purity']:.1%} (代表中被杀死的比例)")
    
    print(f"\n实验组:")
    print(f"  保留杀死数: {eval_exp['n_killed_retained']}/{eval_exp['n_killed_original']}")
    print(f"  MS 保持率: {eval_exp['reduced_ms']:.1%}")
    print(f"  代表纯度: {eval_exp['rep_purity']:.1%}")
    
    # MS 损失分析
    loss_base = 1 - eval_base['reduced_ms']
    loss_exp = 1 - eval_exp['reduced_ms']
    print(f"\n【MS 损失分析】")
    print(f"对照组损失: {loss_base:.1%} (丢弃了 {eval_base['n_killed_original'] - eval_base['n_killed_retained']} 个杀死变异体)")
    print(f"实验组损失: {loss_exp:.1%} (丢弃了 {eval_exp['n_killed_original'] - eval_exp['n_killed_retained']} 个杀死变异体)")
    
    if eval_exp['reduced_ms'] >= eval_base['reduced_ms']:
        print(f"✅ 实验组 MS 保持更优 (多保持 {eval_exp['reduced_ms']-eval_base['reduced_ms']:.1%})")
    else:
        print(f"⚠️ 实验组 MS 损失较大 (少保持 {eval_base['reduced_ms']-eval_exp['reduced_ms']:.1%})")

if __name__ == "__main__":
    main()