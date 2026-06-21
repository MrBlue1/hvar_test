import os
import sys
import importlib.util
import numpy as np
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import csv

# =====================================================
# 基本路径与配置
# =====================================================
mutants_dir = Path(__file__).parent
ORACLE_ID = "M00"

# 完善算子类型映射（基于常见变异算子分类）
MUTANT_OPERATOR_MAP = {
    "M00": "ORIG",
    "M01": "ROR", "M02": "ROR", "M03": "ROR", "M04": "ROR",
    "M05": "ROR", "M06": "ROR", "M07": "ROR", "M08": "ROR",
    "M09": "ROR", "M10": "ROR", "M11": "ROR", "M12": "ROR",
    "M13": "AOR", "M14": "AOR", "M15": "AOR", "M16": "AOR",
    "M17": "AOR", "M18": "AOR", "M19": "AOR", "M20": "AOR",
    "M21": "AOR", "M22": "AOR",
    "M23": "LOR", "M24": "LOR", "M25": "LOR", "M26": "LOR",
    "M27": "LOR", "M28": "LOR", "M29": "LOR", "M30": "LOR",
    "M31": "COR", "M32": "COR", "M33": "COR", "M34": "COR",
    "M35": "COR", "M36": "COR",
    "M37": "UOI", "M38": "UOI", "M39": "UOI", "M40": "UOI",
    "M41": "UOI", "M42": "UOI", "M43": "UOI",
    "M44": "SDL", "M45": "SDL", "M46": "SDL", "M47": "SDL",
    "M48": "SDL", "M49": "SDL", "M50": "SDL",
    "M51": "ABS", "M52": "ABS", "M53": "ABS", "M54": "ABS",
    "M55": "ABS", "M56": "ABS", "M57": "ABS", "M58": "ABS",
    "M59": "ABS", "M60": "ABS",  
    #以下是等价变异体
    "M61": "AOR",  # 幂运算 ** 改为乘法 *
    "M62": "COR",  # 加法交换律（X_norm + Y_norm 交换顺序）
    "M63": "AOR",  # 减法改为加负数（-2.0 → +(-2.0)）
    "M64": "COR",  # 指数项交换律（-gamma*dist_sq + eps → eps - gamma*dist_sq）
    "M65": "COR",  # 对称化加法交换（K + K.T → K.T + K）
    "M66": "AOR",  # 除法改为乘法（/2.0 → *0.5）
    "M67": "UOI",  # 一元操作注入（+0.0, *1.0）
    "M68": "AOR",  # 结合律括号变更（(a+b)-c → a+(b-c) 的括号形式）
    "M69": "SDL",  # 死代码插入（if False 块，等价于无操作）
    "M70": "COR",  # 条件操作强化（if Y is X → if Y is X or True）
    "M71": "ROR",  # 关系/顺序操作（maximum 参数交换）
    "M72": "ROR",  # reshape 维度指定（-1 改为具体数值）
    "M73": "SDL",  # 冗余自赋值（X = temp; temp = X）
    "M74": "AOR",  # 乘法分解（-2.0 → -1.0 * 2.0）
    "M75": "SDL",  # 语句替换（fill_diagonal 改为循环）
}

# =====================================================
# 严格警告抑制
# =====================================================
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# =====================================================
# 行为标签
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
# 行为分类
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
# 行为差异判定（M00 作为 Oracle）
# =====================================================
def behavior_diff(ref, mut, atol=1e-3, rtol=1e-5) -> bool:
    if ref is None or mut is None:
        return True
    if not isinstance(ref, np.ndarray) or not isinstance(mut, np.ndarray):
        return True
    if ref.shape != mut.shape:
        return True
    
    # 检查 NaN 模式是否一致（相同位置都是 NaN 不算差异）
    nan_mask_ref = np.isnan(ref)
    nan_mask_mut = np.isnan(mut)
    if not np.array_equal(nan_mask_ref, nan_mask_mut):
        return True  # NaN 位置不同，判定为差异
    
    # 检查 Inf 模式是否一致
    inf_mask_ref = np.isinf(ref)
    inf_mask_mut = np.isinf(mut)
    if not np.array_equal(inf_mask_ref, inf_mask_mut):
        return True  # Inf 位置不同，判定为差异
    
    # 对非 NaN/Inf 的位置进行数值比较（equal_nan=True 会自动处理 NaN）
    return not np.allclose(ref, mut, atol=atol, rtol=rtol, equal_nan=True)

# =====================================================
# 运行测试，生成覆盖向量和详细信息
# =====================================================
def run_tests_with_detail(oracle_kernel, mutant_kernel, tests):
    """
    运行测试，返回：
    - ms_vector: [n_tests * 2] (0:存活, 1:杀死)
    - detail_vector: [n_tests * 9] (行为分类 one-hot)
    """
    ms_results = []
    detail_results = []

    for X, Y, gamma, eps in tests:
        exception = None
        actual = None
        ref = None
        
        try:
            with np.errstate(all="ignore"):
                ref = oracle_kernel(X, Y, gamma, eps)
                actual = mutant_kernel(X, Y, gamma, eps)
        except Exception as e:
            exception = e

        # MS 判断：与 Oracle 行为不同 = 杀死
        is_killed = behavior_diff(ref, actual) if exception is None else True
        
        v_ms = np.zeros(2)
        v_ms[1 if is_killed else 0] = 1
        ms_results.append(v_ms)
        
        # 细粒度行为分类
        label = classify_behavior(None, actual, X, Y, exception)
        v_detail = np.zeros(len(LABELS))
        v_detail[label2idx[label]] = 1
        detail_results.append(v_detail)

    return {
        'ms': np.concatenate(ms_results),
        'detail': np.concatenate(detail_results)
    }

# =====================================================
# 生成覆盖向量（返回 MS 向量和 Detail 向量）
# =====================================================
def generate_coverage_vectors(n_tests: int, seed: int = 42, progress_callback=None):
    """
    生成所有变异体的行为向量（基于 M00 差异 Oracle）
    """
    tests = lhs_samples(n_tests, seed=seed)
    oracle_kernel = load_kernel_func(mutants_dir / f"{ORACLE_ID}.py")
    
    coverage_vectors = {}
    detail_vectors = {}
    
    mutant_ids = [k for k in MUTANT_OPERATOR_MAP.keys() if k != "M00"]
    
    for i, mid in enumerate(mutant_ids):
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            continue

        mutant_kernel = load_kernel_func(py_file)
        result = run_tests_with_detail(oracle_kernel, mutant_kernel, tests)
        
        coverage_vectors[mid] = result['ms']
        detail_vectors[mid] = result['detail']
        
        if progress_callback:
            progress_callback(i + 1, len(mutant_ids))

    return coverage_vectors, detail_vectors

# =====================================================
# 辅助函数：获取变异体杀死数和行为特征
# =====================================================
def get_mutant_kill_count(behavior_vector: np.ndarray) -> int:
    """获取变异体杀死的测试用例数"""
    if behavior_vector.size % 2 != 0:
        return 0
    n_tests = behavior_vector.size // 2
    bv = behavior_vector.reshape(n_tests, 2)
    return int(np.sum(bv[:, 1] == 1))

def get_mutant_behavior_signature(behavior_vector: np.ndarray, detail_vector: np.ndarray) -> str:
    """获取变异体的主导行为特征"""
    if detail_vector.size % len(LABELS) != 0:
        return "未知"
    
    n_tests = detail_vector.size // len(LABELS)
    dv = detail_vector.reshape(n_tests, len(LABELS))
    behavior_counts = np.sum(dv, axis=0)
    
    # 排除"正确"，找错误类型中最多的
    error_counts = behavior_counts.copy()
    error_counts[label2idx["正确"]] = 0
    
    if np.sum(error_counts) == 0:
        return "正确"
    
    dominant_idx = np.argmax(error_counts)
    return LABELS[dominant_idx]

def get_mutant_behavior_distribution(detail_vector: np.ndarray) -> Dict[str, int]:
    """获取变异体的行为分布统计"""
    if detail_vector.size % len(LABELS) != 0:
        return {}
    
    n_tests = detail_vector.size // len(LABELS)
    dv = detail_vector.reshape(n_tests, len(LABELS))
    behavior_counts = np.sum(dv, axis=0)
    
    return {LABELS[i]: int(behavior_counts[i]) for i in range(len(LABELS))}

def is_mutant_killed_by_vector(behavior_vector: np.ndarray) -> bool:
    """判断变异体是否被杀死（只要有1个测试杀死就算）"""
    if behavior_vector.size % 2 != 0:
        return False
    n_tests = behavior_vector.size // 2
    bv = behavior_vector.reshape(n_tests, 2)
    return np.any(bv[:, 1] == 1)

# =====================================================
# MS 计算函数
# =====================================================
def compute_mutation_score(coverage_vectors: Dict[str, np.ndarray]):
    """计算完整变异分数"""
    killed = 0
    total = len(coverage_vectors)
    
    for vec in coverage_vectors.values():
        if is_mutant_killed_by_vector(vec):
            killed += 1
            
    return {
        "total_mutants": total,
        "killed_mutants": killed,
        "mutation_score": killed / total if total > 0 else 0.0
    }

def compute_reduced_mutation_score(coverage_vectors: Dict[str, np.ndarray],
                                   dict_represent: Dict[str, str]):
    """基于约简代表变异体的 Mutation Score"""
    killed = 0
    total = len(dict_represent)

    for mid in dict_represent.values():
        if mid not in coverage_vectors:
            continue
        if is_mutant_killed_by_vector(coverage_vectors[mid]):
            killed += 1

    return {
        "clusters": total,
        "killed_representatives": killed,
        "reduced_mutation_score": killed / total if total > 0 else 0.0
    }

# =====================================================
# 对照组：分层 KMeans 聚类（基于 kill matrix）
# =====================================================
from sklearn.cluster import KMeans

def stratified_kmeans_baseline(coverage_vectors: Dict[str, np.ndarray], 
                               k_budget: int,
                               operator_map: Dict[str, str]) -> Tuple[Dict[str, str], Dict]:
    """
    对照组：分层 KMeans 聚类（基于杀死模式，非行为指纹）
    """
    # 按算子类型分组
    operator_groups = defaultdict(list)
    for mid, vec in coverage_vectors.items():
        if mid == "M00":
            continue
        op_type = operator_map.get(mid, "UNK")
        operator_groups[op_type].append(mid)
    
    # 计算每组的预算分配
    unique_ops = list(operator_groups.keys())
    n_ops = len(unique_ops)
    
    representatives = {}
    stats = {
        "operator_groups": n_ops,
        "budget_allocation": {},
        "selected_per_group": {}
    }
    
    if k_budget < n_ops:
        # 预算不足，选择 killing rate 最高的 k 个算子类型
        op_kill_rates = {}
        for op in unique_ops:
            group = operator_groups[op]
            total_kills = sum(get_mutant_kill_count(coverage_vectors[m]) for m in group)
            avg_kills = total_kills / len(group) if group else 0
            op_kill_rates[op] = avg_kills
        
        sorted_ops = sorted(op_kill_rates.keys(), key=lambda x: op_kill_rates[x], reverse=True)
        selected_ops = sorted_ops[:k_budget]
        print(f"⚠️ 预算不足(k={k_budget} < 算子数{n_ops})，只覆盖{k_budget}类大组")
        stats["warning"] = "budget_insufficient"
    else:
        selected_ops = unique_ops
        print(f"算子类型数:{n_ops}, 预算k={k_budget}")
    
    # 每组分配的预算
    base_budget = k_budget // len(selected_ops) if selected_ops else 0
    extra = k_budget % len(selected_ops) if selected_ops else 0
    
    allocation = {}
    for i, op in enumerate(selected_ops):
        allocation[op] = base_budget + (1 if i < extra else 0)
    
    stats["budget_allocation"] = allocation
    
    # 对每组进行聚类
    rep_count = 0
    for op in selected_ops:
        group = operator_groups[op]
        group_budget = min(allocation[op], len(group))
        
        if len(group) == 0:
            continue
            
        # 如果预算>=组大小，全选
        if group_budget >= len(group):
            for mid in group:
                representatives[f"cluster_{rep_count}"] = mid
                rep_count += 1
            stats["selected_per_group"][op] = len(group)
            print(f"  {op}: {len(group)}个变异体全选")
            continue
        
        # 构建特征矩阵 [n_mutants, n_tests] - 基于杀死模式
        X = []
        valid_mids = []
        for mid in group:
            if mid in coverage_vectors:
                vec = coverage_vectors[mid].reshape(-1, 2)[:, 1]  # 杀死维度
                X.append(vec)
                valid_mids.append(mid)
        
        if len(valid_mids) == 0:
            continue
            
        X = np.array(X)
        
        # 检查实际独特模式数
        unique_patterns = len(np.unique(X, axis=0))
        actual_clusters = min(group_budget, unique_patterns)
        
        if actual_clusters < group_budget:
            print(f"  ⚠️  {op}: 实际只有{unique_patterns}种独特杀死模式，调整簇数为{actual_clusters}")
            group_budget = actual_clusters
        
        # KMeans聚类
        if unique_patterns == 1:
            representatives[f"cluster_{rep_count}"] = valid_mids[0]
            rep_count += 1
            stats["selected_per_group"][op] = 1
            print(f"  {op}: {len(group)}个变异体只有1种模式，选1个代表")
        else:
            kmeans = KMeans(n_clusters=group_budget, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # 从每个簇选杀死率最高的
            selected = set()
            for cluster_id in range(group_budget):
                cluster_mask = (labels == cluster_id)
                cluster_mids = [valid_mids[i] for i in range(len(valid_mids)) if cluster_mask[i]]
                
                if cluster_mids:
                    best_mid = max(cluster_mids, 
                                  key=lambda m: get_mutant_kill_count(coverage_vectors[m]))
                    representatives[f"cluster_{rep_count}"] = best_mid
                    rep_count += 1
                    selected.add(best_mid)
            
            stats["selected_per_group"][op] = len(selected)
            print(f"  {op}: {len(group)}个变异体分{group_budget}簇（原预算{allocation[op]}），选{len(selected)}个代表")
    
    print(f"【对照组】最终选中 {len(representatives)} 个代表")
    return representatives, stats

# =====================================================
# 【修复】实验组：算子优先 + 行为指纹聚类
# =====================================================
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def operator_priority_behavior_clustering(coverage_vectors: Dict[str, np.ndarray],
                                         detail_vectors: Dict[str, np.ndarray],
                                         k_budget: int,
                                         operator_map: Dict[str, str]) -> Tuple[Dict[str, str], Dict]:
    """
    【修复版】实验组：算子优先 + 行为指纹聚类
    
    核心改进：
    1. 按算子类型分组（算子优先）
    2. 组内基于行为指纹（detail_vectors）进行聚类
    3. 从每个行为簇中选择kill count最高的代表
    4. 剩余预算按kill count填充
    """
    # 按算子类型分组
    operator_groups = defaultdict(list)
    kill_counts = {}
    
    for mid, vec in coverage_vectors.items():
        if mid == "M00":
            continue
        op_type = operator_map.get(mid, "UNK")
        operator_groups[op_type].append(mid)
        kill_counts[mid] = get_mutant_kill_count(vec)
    
    # 获取存活的变异体（被杀死的）
    killed_mutants = [m for m in coverage_vectors.keys() 
                      if m != "M00" and is_mutant_killed_by_vector(coverage_vectors[m])]
    n_killed = len(killed_mutants)
    total_mutants = len([m for m in coverage_vectors.keys() if m != "M00"])
    
    print(f"【算子优先+行为指纹聚类】总变异体:{total_mutants}, 被杀死的:{n_killed}, 预算k={k_budget}")
    
    representatives = {}
    stats = {
        "operators_covered": [],
        "behavior_clusters": {},
        "killed_selected": 0,
        "fill_count": 0
    }
    
    unique_ops = list(operator_groups.keys())
    n_ops = len(unique_ops)
    
    # 第一步：确定覆盖哪些算子类型
    if k_budget < n_ops:
        print(f"⚠️ 预算紧张(k={k_budget} < 算子数{n_ops})，只覆盖{k_budget}类")
        # 选择杀死率最高的k个算子类型
        op_kill_rates = {}
        for op in unique_ops:
            group = operator_groups[op]
            total_kills = sum(kill_counts.get(m, 0) for m in group)
            avg_kills = total_kills / len(group) if group else 0
            op_kill_rates[op] = avg_kills
        
        sorted_ops = sorted(op_kill_rates.keys(), key=lambda x: op_kill_rates[x], reverse=True)
        selected_ops = sorted_ops[:k_budget]
        print(f"  选中算子类型: {selected_ops}")
    else:
        print(f"算子类型数:{n_ops}类, 预算k={k_budget}")
        print(f"✓ 预算充足(k={k_budget} >= 算子数{n_ops})，可覆盖所有算子类型")
        selected_ops = unique_ops
    
    # 计算每组的预算分配
    base_budget = k_budget // len(selected_ops) if selected_ops else 0
    extra = k_budget % len(selected_ops) if selected_ops else 0
    
    allocation = {}
    for i, op in enumerate(selected_ops):
        allocation[op] = base_budget + (1 if i < extra else 0)
    
    # 第二步：对每个算子组进行行为指纹聚类
    rep_count = 0
    covered_ops = []
    
    for op in selected_ops:
        group = operator_groups[op]
        group_budget = min(allocation[op], len(group))
        
        if not group or group_budget <= 0:
            continue
        
        covered_ops.append(op)
        
        # 如果预算>=组大小，全选
        if group_budget >= len(group):
            for mid in group:
                representatives[f"cluster_{rep_count}"] = mid
                rep_count += 1
            stats["behavior_clusters"][op] = len(group)
            print(f"  {op}: {len(group)}个变异体全选（预算充足）")
            continue
        
        # 【关键】构建行为指纹特征矩阵 [n_mutants, n_tests * n_labels]
        behavior_features = []
        valid_mids = []
        
        for mid in group:
            if mid in detail_vectors:
                # detail_vector: [n_tests * n_labels] - 每个测试用例的行为one-hot
                dv = detail_vectors[mid]
                behavior_features.append(dv)
                valid_mids.append(mid)
        
        if len(valid_mids) == 0:
            continue
        
        X_behavior = np.array(behavior_features)
        
        # 【关键】降维处理（行为指纹维度可能很高：n_tests * 9）
        n_features = X_behavior.shape[1]
        if n_features > 50:
            # 使用PCA降维，保留主要行为模式
            n_components = min(20, group_budget * 2, n_features)
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X_behavior)
            explained_var = sum(pca.explained_variance_ratio_) * 100
            dim_info = f"PCA降维{n_features}→{n_components}维(解释{explained_var:.1f}%方差)"
        else:
            X_reduced = X_behavior
            dim_info = f"原始维度{n_features}"
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        # 【关键】基于行为相似性进行KMeans聚类
        unique_patterns = len(np.unique(X_scaled, axis=0))
        actual_clusters = min(group_budget, unique_patterns)
        
        if actual_clusters < group_budget:
            print(f"  ⚠️  {op}: {dim_info}，实际只有{unique_patterns}种独特行为模式，调整簇数为{actual_clusters}")
            group_budget = actual_clusters
        
        if unique_patterns == 1:
            # 只有一种行为模式，选kill count最高的
            best_mid = max(valid_mids, key=lambda m: kill_counts.get(m, 0))
            representatives[f"cluster_{rep_count}"] = best_mid
            rep_count += 1
            stats["behavior_clusters"][op] = 1
            
            behavior = get_mutant_behavior_signature(coverage_vectors[best_mid], detail_vectors[best_mid])
            print(f"  {op}: 1种行为模式，选{best_mid}(行为:{behavior}, 杀死:{kill_counts.get(best_mid, 0)})")
        else:
            # KMeans基于行为指纹聚类
            kmeans = KMeans(n_clusters=group_budget, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # 从每个行为簇中选择kill count最高的代表
            selected = set()
            behavior_dist = defaultdict(int)
            
            for cluster_id in range(group_budget):
                cluster_mask = (labels == cluster_id)
                cluster_mids = [valid_mids[i] for i in range(len(valid_mids)) if cluster_mask[i]]
                
                if cluster_mids:
                    # 在该行为簇内选kill count最高的
                    best_mid = max(cluster_mids, 
                                  key=lambda m: kill_counts.get(m, 0))
                    representatives[f"cluster_{rep_count}"] = best_mid
                    rep_count += 1
                    selected.add(best_mid)
                    
                    # 记录该簇的主导行为
                    behavior = get_mutant_behavior_signature(
                        coverage_vectors[best_mid], detail_vectors[best_mid]
                    )
                    behavior_dist[behavior] += 1
            
            stats["behavior_clusters"][op] = len(selected)
            behavior_summary = ", ".join([f"{b}:{c}" for b, c in behavior_dist.items()])
            print(f"  {op}: {dim_info}，分{group_budget}个行为簇，选{len(selected)}个代表")
            print(f"    行为分布: {behavior_summary}")
    
    stats["operators_covered"] = covered_ops
    
    # 第三步：如果还有剩余预算，补充高杀死率的变异体
    remaining_budget = k_budget - len(representatives)
    
    if remaining_budget > 0:
        selected_mids = set(representatives.values())
        
        # 从未选的且被杀死的变异体中按杀死率排序
        candidates = []
        for mid in killed_mutants:
            if mid not in selected_mids:
                op = operator_map.get(mid, "UNK")
                behavior = get_mutant_behavior_signature(coverage_vectors[mid], detail_vectors[mid])
                candidates.append((mid, kill_counts.get(mid, 0), op, behavior))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n【预算填充】剩余{remaining_budget}个名额，按杀死率补充:")
        
        for i in range(min(remaining_budget, len(candidates))):
            mid, kills, op, behavior = candidates[i]
            representatives[f"cluster_{rep_count}"] = mid
            print(f"  + {mid} | 算子:{op} | 行为:{behavior} | 杀死:{kills}")
            rep_count += 1
            stats["fill_count"] += 1
    
    print(f"\n【最终结果】选中{len(representatives)}个代表（预算k={k_budget}）")
    print(f"  覆盖算子类型: {len(covered_ops)}/{n_ops} - {sorted(covered_ops)}")
    print(f"  行为簇统计: {dict(stats['behavior_clusters'])}")
    print(f"  ✅ 预算使用正好：{len(representatives)} = k={k_budget}")
    
    return representatives, stats

# =====================================================
# FTRR 计算
# =====================================================
def compute_ftrr(original_operators: Set[str], selected_operators: Set[str]) -> Tuple[float, str]:
    """
    计算故障类型保留率 (Fault Type Retention Rate)
    FTRR = |选中算子类型 ∩ 原始算子类型| / |原始算子类型|
    """
    if not original_operators:
        return 0.0, "0/0"
    
    retained = len(original_operators & selected_operators)
    total = len(original_operators)
    ftrr = retained / total if total > 0 else 0.0
    detail = f"{retained}/{total}"
    return ftrr, detail

# =====================================================
# 二维实验框架
# =====================================================
def run_2d_experiment(test_sizes=[20, 50, 100, 200], 
                     budgets=[5, 10, 20, 25],
                     seed=42):
    """
    运行二维实验：测试用例规模 × 预算约束
    """
    print("=" * 70)
    print("RBF Kernel - 算子优先+行为指纹聚类 二维实验")
    print("=" * 70)
    print(f"测试规模: {test_sizes}")
    print(f"预算取值: {budgets}")
    print("=" * 70)
    
    results = []
    
    for n_tests in test_sizes:
        print(f"\n{'='*70}")
        print(f"【测试用例规模 N = {n_tests}】")
        print(f"{'='*70}")
        print(f"生成 {n_tests} 个 LHS 测试用例...")
        
        # 生成测试用例和运行所有变异体
        def progress(current, total):
            if current % 10 == 0 or current == total:
                print(f"  进度: {current}/{total} 变异体")
        
        coverage_vectors, detail_vectors = generate_coverage_vectors(n_tests, seed, progress)
        
        # 诊断信息
        all_mutants = [m for m in coverage_vectors.keys() if m != "M00"]
        killed_mutants = [m for m in all_mutants if is_mutant_killed_by_vector(coverage_vectors[m])]
        print(f"\n所有测试完成，共处理 {len(all_mutants)} 个变异体")
        print(f"诊断: 被杀死变异体 = {len(killed_mutants)}/{len(all_mutants)}, "
              f"Kill Matrix 形状 = ({len(all_mutants)}, {n_tests})")
        
        # 获取原始算子类型集合
        original_operators = set()
        for mid in all_mutants:
            original_operators.add(MUTANT_OPERATOR_MAP.get(mid, "UNK"))
        
        for k in budgets:
            print(f"\n  [K={k}]")
            
            # 对照组：分层KMeans（基于kill matrix）
            print("【对照组-分层KMeans】", end="")
            baseline_reps, baseline_stats = stratified_kmeans_baseline(
                coverage_vectors, k, MUTANT_OPERATOR_MAP
            )
            
            # 计算对照组FTRR
            baseline_ops = set()
            for mid in baseline_reps.values():
                baseline_ops.add(MUTANT_OPERATOR_MAP.get(mid, "UNK"))
            baseline_ftrr, baseline_ftrr_detail = compute_ftrr(original_operators, baseline_ops)
            print(f"  [Debug] 原始算子类型: {sorted(original_operators)}")
            print(f"  [Debug] 代表算子类型: {sorted(baseline_ops)}")
            print(f"  [Debug] FTRR计算: {baseline_ftrr_detail}={baseline_ftrr*100:.1f}%")
            
            # 计算对照组MS
            baseline_ms_result = compute_reduced_mutation_score(coverage_vectors, baseline_reps)
            baseline_ms = baseline_ms_result['reduced_mutation_score'] * 100
            
            # 实验组：算子优先+行为指纹聚类
            print()
            proposed_reps, proposed_stats = operator_priority_behavior_clustering(
                coverage_vectors, detail_vectors, k, MUTANT_OPERATOR_MAP
            )
            
            # 计算实验组FTRR
            proposed_ops = set()
            for mid in proposed_reps.values():
                proposed_ops.add(MUTANT_OPERATOR_MAP.get(mid, "UNK"))
            proposed_ftrr, proposed_ftrr_detail = compute_ftrr(original_operators, proposed_ops)
            print(f"  [Debug] 原始算子类型: {sorted(original_operators)}")
            print(f"  [Debug] 代表算子类型: {sorted(proposed_ops)}")
            print(f"  [Debug] FTRR计算: {proposed_ftrr_detail}={proposed_ftrr*100:.1f}%")
            
            # 计算实验组MS
            proposed_ms_result = compute_reduced_mutation_score(coverage_vectors, proposed_reps)
            proposed_ms = proposed_ms_result['reduced_mutation_score'] * 100
            
            # 计算提升
            ms_improvement = proposed_ms - baseline_ms
            ftrr_improvement = (proposed_ftrr - baseline_ftrr) * 100
            
            print(f"\nMS: {baseline_ms:.1f}% → {proposed_ms:.1f}% ({ms_improvement:+.1f}%) | "
                  f"FTRR: {baseline_ftrr*100:.1f}% vs {proposed_ftrr*100:.1f}%")
            
            # 记录结果
            results.append({
                'n_tests': n_tests,
                'k': k,
                'baseline_ms': baseline_ms,
                'proposed_ms': proposed_ms,
                'ms_improvement': ms_improvement,
                'baseline_ftrr': baseline_ftrr * 100,
                'proposed_ftrr': proposed_ftrr * 100,
                'ftrr_improvement': ftrr_improvement
            })
    
    # 打印汇总表格
    print("\n" + "=" * 70)
    print("【实验结果汇总】")
    print("=" * 70)
    
    current_n = None
    for r in results:
        if r['n_tests'] != current_n:
            current_n = r['n_tests']
            print(f"\n测试用例数 = {current_n}:")
            print(f"{'K':<5} {'Baseline MS':<12} {'Proposed MS':<12} {'提升':<12} {'FTRR 提升':<12}")
            print("-" * 60)
        
        print(f"{r['k']:<5} {r['baseline_ms']:.1f}%{'':<6} {r['proposed_ms']:.1f}%{'':<6} "
              f"{r['ms_improvement']:+.1f}%{'':<6} {r['ftrr_improvement']:+.1f}%")
    
    # 保存到CSV
    csv_file = f"rbf_behavior_clustering_seed{seed}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_tests', 'k', 'baseline_ms', 'proposed_ms', 
                                               'ms_improvement', 'baseline_ftrr', 'proposed_ftrr', 
                                               'ftrr_improvement'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n结果已保存到: {csv_file}")
    
    # 关键发现
    print("\n" + "=" * 70)
    print("【关键发现】")
    print("=" * 70)
    max_improvement = max(results, key=lambda x: x['ms_improvement'])
    print(f"最大 MS 提升: +{max_improvement['ms_improvement']:.1f}% "
          f"(N={max_improvement['n_tests']}, K={max_improvement['k']})")
    
    # 分析行为聚类的优势
    print(f"\n【方法对比】")
    print(f"对照组: 分层KMeans（基于kill matrix相似性）")
    print(f"实验组: 算子优先+行为指纹聚类（基于行为模式相似性）")
    print(f"核心差异: 实验组在算子组内按'行为类型'而非'杀死模式'聚类")
    
    return results

# =====================================================
# 主入口
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RBF Kernel - 算子优先+行为指纹聚类 实验框架")
    print("=" * 60)
    
    # 运行二维实验
    run_2d_experiment(
        test_sizes=[20, 50, 100, 200],
        budgets=[5, 10, 20, 25],
        seed=42
    )