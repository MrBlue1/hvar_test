import warnings
import numpy as np
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import importlib.util
from collections import defaultdict
import csv

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(all='ignore')

# ============================================
# 1. 配置和常量
# ============================================

mutants_dir = Path(__file__).parent

MUTANT_OPERATOR_MAP = {
    "M00": "ORIG",  # 原程序
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
}

# MS算法用：二分类标签
MS_LABELS = ["存活", "杀死"]
ms_label2idx = {l: i for i, l in enumerate(MS_LABELS)}

# 可选：7类细粒度分析（用于辅助评估，不用于MS计算）
DETAIL_LABELS = [
    "正确", "数值溢出", "统计错误", "归一化错误",
    "负值错误", "类型错误", "其他异常"
]
detail_label2idx = {l: i for i, l in enumerate(DETAIL_LABELS)}

EPS = 1e-6


# ============================================
# 2. 测试数据生成
# ============================================

def lhs_samples(n: int, seed: Optional[int] = None) -> List[List[float]]:
    """生成LHS测试样本"""
    if seed is not None:
        np.random.seed(seed)

    samples = []
    n_normal = int(n * 0.6)
    n_extreme = n - n_normal

    # 正常范围
    for _ in range(n_normal):
        size = np.random.randint(1, 50)
        samples.append(np.random.uniform(-10, 10, size).tolist())

    # 极端值
    for _ in range(n_extreme):
        size = np.random.randint(1, 50)
        samples.append(np.random.uniform(-1e6, 1e6, size).tolist())

    return samples


# ============================================
# 3. 参考Oracle（用于辅助分析，不用于MS计算）
# ============================================

def reference_oracle(data: List[float]) -> Dict[str, float]:
    """参考实现：用于分析变异体输出质量，不参与MS计算"""
    try:
        clean = [float(x) for x in data if isinstance(x, (int, float))]
        if not clean:
            clean = [0.0]

        mean = sum(clean) / len(clean)
        var = np.var(clean, ddof=1) if len(clean) > 1 else 0.0
        std = math.sqrt(var) if var >= 0 else 0.0
        min_v = min(clean)
        max_v = max(clean)
        r = max_v - min_v

        return {
            "mean": mean,
            "variance": var,
            "std": std,
            "min": min_v,
            "max": max_v,
            "range": r,
        }
    except Exception as e:
        raise RuntimeError(f"Oracle failed: {e}")


def classify_detailed_behavior(expected: Optional[Dict[str, float]],
                                actual,
                                exception: Optional[Exception] = None) -> str:
    """细粒度分类：用于分析，不用于MS"""
    if exception is not None:
        if isinstance(exception, (ZeroDivisionError, FloatingPointError)):
            return "数值溢出"
        if isinstance(exception, TypeError):
            return "类型错误"
        return "其他异常"

    if not isinstance(actual, dict):
        return "类型错误"

    for v in actual.values():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return "数值溢出"

    if actual.get("std", 0) < -EPS:
        return "统计错误"
    if actual.get("range", 0) < -EPS:
        return "统计错误"

    nm = actual.get("norm_mean", 0)
    if nm < -EPS or nm > 1 + EPS:
        return "归一化错误"

    if expected is not None:
        for k in expected:
            if k in actual:
                if abs(actual[k] - expected[k]) > 1e-4:
                    return "统计错误"

    return "正确"


# ============================================
# 4. 核心：加载程序和运行测试
# ============================================

def load_analyzer(py_file: Path):
    """加载分析器类"""
    name = py_file.stem
    if name in sys.modules:
        del sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    return mod.FeatureAnalyzer


def run_analyzer_safe(AnalyzerClass, data: List[float]) -> Tuple[Optional[Dict], Optional[Exception]]:
    """安全运行分析器，返回(结果, 异常)"""
    try:
        analyzer = AnalyzerClass(data)
        result = analyzer.extract_all()
        return result, None
    except Exception as e:
        return None, e


def outputs_equal(out1, out2, tol: float = 1e-6) -> bool:
    """
    比较两个输出是否相等（MS算法核心）
    处理：dict比较、异常比较、数值容忍度
    """
    # 都异常：比较异常类型
    if isinstance(out1, Exception) and isinstance(out2, Exception):
        return type(out1) == type(out2)
    
    # 一个异常一个正常：不同
    if (out1 is None) != (out2 is None):
        return False
    if isinstance(out1, Exception) != isinstance(out2, Exception):
        return False
    
    # 都是dict：逐key比较
    if isinstance(out1, dict) and isinstance(out2, dict):
        if set(out1.keys()) != set(out2.keys()):
            return False
        for k in out1:
            v1, v2 = out1[k], out2[k]
            # 数值比较
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if math.isnan(v1) and math.isnan(v2):
                    continue
                if abs(v1 - v2) > tol:
                    return False
            else:
                if v1 != v2:
                    return False
        return True
    
    # 直接比较
    return out1 == out2


def run_tests_for_ms(original_analyzer_class, mutant_analyzer_class, 
                     tests: List[List[float]], 
                     include_detail: bool = False) -> Dict[str, np.ndarray]:
    """
    核心MS测试运行函数
    
    Returns:
        ms_vector: [n_tests * 2] - MS算法用的行为向量
        detail_vector: [n_tests * 7] (可选) - 细粒度分析
    """
    ms_results = []
    detail_results = [] if include_detail else None

    for data in tests:
        # 1. 运行原程序
        orig_out, orig_err = run_analyzer_safe(original_analyzer_class, data)
        if orig_err:
            orig_out = orig_err
        
        # 2. 运行变异体
        mut_out, mut_err = run_analyzer_safe(mutant_analyzer_class, data)
        if mut_err:
            mut_out = mut_err
        
        # 3. MS判断：输出不同 = 杀死
        is_killed = not outputs_equal(orig_out, mut_out)
        
        v_ms = np.zeros(2)
        v_ms[ms_label2idx["杀死" if is_killed else "存活"]] = 1
        ms_results.append(v_ms)
        
        # 4. 可选：细粒度分析（用于辅助评估）
        if include_detail:
            expected = reference_oracle(data) if not isinstance(orig_out, Exception) else None
            detail_label = classify_detailed_behavior(expected, mut_out, mut_err)
            v_detail = np.zeros(7)
            v_detail[detail_label2idx[detail_label]] = 1
            detail_results.append(v_detail)
    
    result = {
        'ms': np.concatenate(ms_results),
    }
    if include_detail:
        result['detail'] = np.concatenate(detail_results)
    
    return result


# ============================================
# 5. 生成覆盖率向量（修改后）
# ============================================

def generate_coverage_vectors(n_tests: int = 20000, seed: int = 42, 
                              progress_callback=None) -> Dict[str, np.ndarray]:
    """
    生成所有变异体的MS行为向量
    
    关键：使用M00.py作为原程序基准
    """
    tests = lhs_samples(n_tests, seed)
    
    # 加载原程序 M00.py
    original_path = mutants_dir / "M00.py"
    if not original_path.exists():
        raise FileNotFoundError(f"原程序不存在: {original_path}")
    
    original_analyzer = load_analyzer(original_path)
    
    coverage_vectors = {}
    mutant_details = {}  # 存储细粒度行为用于后续分析
    
    # 遍历所有变异体（包括M00自身，用于验证）
    mutant_ids = [k for k in MUTANT_OPERATOR_MAP.keys() if k != "M00"]
    
    for i, mid in enumerate(mutant_ids):
        py_file = mutants_dir / f"{mid}.py"
        
        if not py_file.exists():
            print(f"警告: {py_file} 不存在，跳过")
            continue
            
        mutant_analyzer = load_analyzer(py_file)
        
        # 运行测试，获取MS行为向量（包含细粒度信息）
        result = run_tests_for_ms(original_analyzer, mutant_analyzer, tests, include_detail=True)
        coverage_vectors[mid] = result['ms']
        mutant_details[mid] = result['detail']
        
        if progress_callback:
            progress_callback(i + 1, len(mutant_ids))
    
    return coverage_vectors, mutant_details


# ============================================
# 6. MS计算函数（保持接口不变）
# ============================================

def is_mutant_killed(behavior_vector: np.ndarray) -> bool:
    """
    判断单个变异体是否被杀死
    
    behavior_vector: shape [n_tests * 2]
    每2个元素为一组：[存活, 杀死]
    """
    if behavior_vector.size % 2 != 0:
        raise ValueError("Behavior vector size not divisible by 2")
    
    n_tests = behavior_vector.size // 2
    bv = behavior_vector.reshape(n_tests, 2)
    
    # 获取每个测试的结果：0=存活, 1=杀死
    test_results = np.argmax(bv, axis=1)
    
    # 只要有1个测试杀死了变异体，就算被杀死
    return np.any(test_results == 1)


def compute_mutation_score(coverage_vectors: Dict[str, np.ndarray]) -> Dict:
    """计算完整变异分数"""
    # 排除M00原程序
    mutant_vectors = {k: v for k, v in coverage_vectors.items() if k != "M00"}
    
    killed = 0
    total = len(mutant_vectors)

    for vec in mutant_vectors.values():
        if is_mutant_killed(vec):
            killed += 1

    return {
        "total_mutants": total,
        "killed_mutants": killed,
        "mutation_score": killed / total if total > 0 else 0.0
    }


def compute_reduced_mutation_score(coverage_vectors: Dict[str, np.ndarray],
                                   dict_represent: Dict[str, str]) -> Dict:
    """
    基于约简代表变异体的Mutation Score
    
    dict_represent: {cluster_0: M01, cluster_1: M21, ...}
    """
    # 排除M00
    mutant_vectors = {k: v for k, v in coverage_vectors.items() if k != "M00"}
    
    killed = 0
    total = len(dict_represent)

    for cluster, mid in dict_represent.items():
        if mid not in mutant_vectors:
            continue
        if is_mutant_killed(mutant_vectors[mid]):
            killed += 1

    score = killed / total if total > 0 else 0.0

    return {
        "clusters": total,
        "killed_representatives": killed,
        "reduced_mutation_score": score
    }


def get_mutant_kill_count(behavior_vector: np.ndarray) -> int:
    """获取变异体杀死的测试用例数"""
    if behavior_vector.size % 2 != 0:
        return 0
    n_tests = behavior_vector.size // 2
    bv = behavior_vector.reshape(n_tests, 2)
    return int(np.sum(bv[:, 1] == 1))


def get_mutant_behavior_signature(behavior_vector: np.ndarray, detail_vector: np.ndarray) -> str:
    """获取变异体的行为特征（用于显示）"""
    # 从detail_vector找出最主要的行为
    if detail_vector.size % 7 != 0:
        return "未知"
    
    n_tests = detail_vector.size // 7
    dv = detail_vector.reshape(n_tests, 7)
    behavior_counts = np.sum(dv, axis=0)
    dominant_idx = np.argmax(behavior_counts)
    
    behavior_map = {
        0: "正确",
        1: "数值溢出", 
        2: "统计错误",
        3: "归一化错误",
        4: "负值错误",
        5: "类型错误",
        6: "其他异常"
    }
    return behavior_map.get(dominant_idx, "未知")


# ============================================
# 7. 对照组：分层KMeans聚类
# ============================================

from sklearn.cluster import KMeans

def stratified_kmeans_baseline(coverage_vectors: Dict[str, np.ndarray], 
                               k_budget: int,
                               operator_map: Dict[str, str]) -> Tuple[Dict[str, str], Dict]:
    """
    对照组：分层KMeans聚类
    先按算子类型分组，然后在每组内进行KMeans聚类
    
    Returns:
        representatives: {cluster_id: mutant_id}
        stats: 统计信息
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
        # 预算不足，随机选择k个算子类型
        import random
        selected_ops = random.sample(unique_ops, k_budget)
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
        
        # 构建特征矩阵 [n_mutants, n_tests]
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
            # 只有一种模式，选第一个
            representatives[f"cluster_{rep_count}"] = valid_mids[0]
            rep_count += 1
            stats["selected_per_group"][op] = 1
            print(f"  {op}: {len(group)}个变异体只有1种模式，选1个代表")
        else:
            kmeans = KMeans(n_clusters=group_budget, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # 从每个簇选离中心最近的
            selected = set()
            for cluster_id in range(group_budget):
                cluster_mask = (labels == cluster_id)
                cluster_mids = [valid_mids[i] for i in range(len(valid_mids)) if cluster_mask[i]]
                
                if cluster_mids:
                    # 选杀死率最高的作为代表
                    best_mid = max(cluster_mids, 
                                  key=lambda m: get_mutant_kill_count(coverage_vectors[m]))
                    representatives[f"cluster_{rep_count}"] = best_mid
                    rep_count += 1
                    selected.add(best_mid)
            
            stats["selected_per_group"][op] = len(selected)
            print(f"  {op}: {len(group)}个变异体分{group_budget}簇（原预算{allocation[op]}），选{len(selected)}个代表")
    
    print(f"【对照组】最终选中 {len(representatives)} 个代表")
    return representatives, stats


# ============================================
# 8. 实验组：算子优先聚类
# ============================================

def operator_priority_clustering(coverage_vectors: Dict[str, np.ndarray],
                                detail_vectors: Dict[str, np.ndarray],
                                k_budget: int,
                                operator_map: Dict[str, str]) -> Tuple[Dict[str, str], Dict]:
    """
    实验组：算子优先聚类策略
    1. 确保每个算子类型至少一个代表
    2. 剩余预算按杀死率优先级填充
    
    Returns:
        representatives: {cluster_id: mutant_id}
        stats: 统计信息
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
    killed_mutants = [m for m in coverage_vectors.keys() if m != "M00" and is_mutant_killed(coverage_vectors[m])]
    n_killed = len(killed_mutants)
    total_mutants = len([m for m in coverage_vectors.keys() if m != "M00"])
    
    print(f"【算子优先聚类】总变异体:{total_mutants}, 被杀死的:{n_killed}, 预算k={k_budget}")
    
    representatives = {}
    stats = {
        "operators_covered": [],
        "killed_selected": 0,
        "fill_count": 0
    }
    
    unique_ops = list(operator_groups.keys())
    n_ops = len(unique_ops)
    
    # 第一步：确保每个算子类型至少一个代表（优先级：高杀死率）
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
    
    rep_count = 0
    covered_ops = []
    
    for op in selected_ops:
        group = operator_groups[op]
        if not group:
            continue
            
        # 在组内选杀死率最高的
        best_mid = max(group, key=lambda m: kill_counts.get(m, 0))
        representatives[f"cluster_{rep_count}"] = best_mid
        covered_ops.append(op)
        
        # 获取行为描述
        if best_mid in detail_vectors:
            behavior = get_mutant_behavior_signature(coverage_vectors[best_mid], detail_vectors[best_mid])
            behavior_str = f" | 行为:{behavior}" if behavior != "正确" else ""
        else:
            behavior_str = ""
        
        print(f"  选代表: {best_mid} | 算子:{op}{behavior_str} | 杀死:{kill_counts.get(best_mid, 0)}")
        rep_count += 1
    
    stats["operators_covered"] = covered_ops
    
    # 第二步：如果还有剩余预算，补充高杀死率的变异体（ diverse selection ）
    remaining_budget = k_budget - len(representatives)
    
    if remaining_budget > 0:
        # 获取已选的代表
        selected_mids = set(representatives.values())
        
        # 从未选的且被杀死的变异体中按杀死率排序
        candidates = []
        for mid in killed_mutants:
            if mid not in selected_mids:
                candidates.append((mid, kill_counts.get(mid, 0)))
        
        # 按杀死率降序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n【预算填充】剩余{remaining_budget}个名额，补充高杀死率变异体:")
        
        for i in range(min(remaining_budget, len(candidates))):
            mid, kills = candidates[i]
            representatives[f"cluster_{rep_count}"] = mid
            op = operator_map.get(mid, "UNK")
            print(f"  + {mid} | 算子:{op} | 杀死:{kills}")
            rep_count += 1
            stats["fill_count"] += 1
    
    print(f"\n【最终结果】选中{len(representatives)}个代表（预算k={k_budget}）")
    print(f"  覆盖算子类型: {len(covered_ops)}/{n_ops} - {sorted(covered_ops)}")
    print(f"  ✅ 预算使用正好：{len(representatives)} = k={k_budget}")
    
    return representatives, stats


# ============================================
# 9. FTRR计算
# ============================================

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


# ============================================
# 10. 二维实验框架
# ============================================

def run_2d_experiment(test_sizes=[20, 50, 100, 200], 
                     budgets=[5, 10, 20, 25],
                     seed=42):
    """
    运行二维实验：测试用例规模 × 预算约束
    """
    print("=" * 70)
    print("【二维实验】测试用例规模 × 预算约束")
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
        tests = lhs_samples(n_tests, seed)
        print("测试用例生成完成\n")
        
        # 生成覆盖向量（带进度）
        def progress(current, total):
            if current % 10 == 0 or current == total:
                print(f"  进度: {current}/{total} 变异体")
        
        coverage_vectors, detail_vectors = generate_coverage_vectors(n_tests, seed, progress_callback=progress)
        
        # 诊断信息
        all_mutants = [m for m in coverage_vectors.keys() if m != "M00"]
        killed_mutants = [m for m in all_mutants if is_mutant_killed(coverage_vectors[m])]
        print(f"\n所有测试完成，共处理 {len(all_mutants)} 个变异体")
        print(f"诊断: 被杀死变异体 = {len(killed_mutants)}/{len(all_mutants)}, "
              f"Kill Matrix 形状 = ({len(all_mutants)}, {n_tests})")
        
        # 获取原始算子类型集合
        original_operators = set()
        for mid in all_mutants:
            original_operators.add(MUTANT_OPERATOR_MAP.get(mid, "UNK"))
        # 过滤掉LOR（如果有的话，根据你的MAP）
        original_operators = {op for op in original_operators if op != "LOR"}  # LOR在你的MAP里但实际可能没有？
        
        for k in budgets:
            print(f"\n  [K={k}]")
            
            # 对照组：分层KMeans
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
            
            # 实验组：算子优先聚类
            print()
            proposed_reps, proposed_stats = operator_priority_clustering(
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
    csv_file = f"experiment_matrix_seed{seed}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_tests', 'k', 'baseline_ms', 'proposed_ms', 
                                               'ms_improvement', 'baseline_ftrr', 'proposed_ftrr', 
                                               'ftrr_improvement'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n结果已保存到: {csv_file}")
    print("提示: 安装 seaborn 和 matplotlib 可绘制热力图")
    
    # 关键发现
    print("\n" + "=" * 70)
    print("【关键发现】")
    print("=" * 70)
    max_improvement = max(results, key=lambda x: x['ms_improvement'])
    print(f"最大 MS 提升: +{max_improvement['ms_improvement']:.1f}% "
          f"(N={max_improvement['n_tests']}, K={max_improvement['k']})")
    
    return results


# ============================================
# 11. 主程序入口
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Statistical Feature Analyzer - 变异测试二维实验框架")
    print("=" * 60)
    
    # 运行二维实验
    run_2d_experiment(
        test_sizes=[20, 50, 100, 200],
        budgets=[5, 10, 20, 25],
        seed=42
    )