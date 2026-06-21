import warnings
import numpy as np
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import importlib.util

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

def generate_coverage_vectors(n_tests: int = 20000, seed: int = 42) -> Dict[str, np.ndarray]:
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
    
    # 遍历所有变异体（包括M00自身，用于验证）
    for mid in MUTANT_OPERATOR_MAP:
        if mid == "M00":
            # 原程序与自身对比，应该全部存活（用于验证）
            py_file = original_path
        else:
            py_file = mutants_dir / f"{mid}.py"
        
        if not py_file.exists():
            print(f"警告: {py_file} 不存在，跳过")
            continue
            
        mutant_analyzer = load_analyzer(py_file)
        
        # 运行测试，获取MS行为向量
        result = run_tests_for_ms(original_analyzer, mutant_analyzer, tests)
        coverage_vectors[mid] = result['ms']
    
    return coverage_vectors


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


def compute_test_averaged_ms(coverage_vectors: Dict[str, np.ndarray]) -> Dict:
    """
    每个test能杀死多少比例的变异体
    
    稳健性指标
    """
    # 排除M00
    mutant_vectors = {k: v for k, v in coverage_vectors.items() if k != "M00"}
    
    if not mutant_vectors:
        return {"error": "No mutant data"}
    
    # 从任意一个mutant推断n_tests
    any_vec = next(iter(mutant_vectors.values()))
    if any_vec.size % 2 != 0:
        raise ValueError("Behavior vector size not divisible by 2")
    
    n_tests = any_vec.size // 2
    n_mutants = len(mutant_vectors)
    
    # 计算每个测试杀死的变异体数
    killed_per_test = np.zeros(n_tests)
    
    for vec in mutant_vectors.values():
        if vec.size != any_vec.size:
            raise ValueError("Inconsistent behavior vector sizes")
        
        bv = vec.reshape(n_tests, 2)
        # 该测试是否杀死此变异体
        is_killed = (bv[:, 1] == 1)  # 杀死列=1
        killed_per_test += is_killed.astype(int)
    
    # 比例
    killed_ratio = killed_per_test / n_mutants

    return {
        "n_tests": n_tests,
        "mean_test_ms": float(np.mean(killed_ratio)),
        "std_test_ms": float(np.std(killed_ratio)),
        "min_test_ms": float(np.min(killed_ratio)),
        "max_test_ms": float(np.max(killed_ratio)),
    }


# ============================================
# 7. 辅助：带细粒度分析的增强版本（可选）
# ============================================

def generate_detailed_analysis(n_tests: int = 1000, seed: int = 42) -> Dict[str, Dict]:
    """
    生成细粒度分析报告（用于论文分析，非MS计算）
    """
    tests = lhs_samples(n_tests, seed)
    
    original_path = mutants_dir / "M00.py"
    original_analyzer = load_analyzer(original_path)
    
    analysis = {}
    
    for mid in MUTANT_OPERATOR_MAP:
        if mid == "M00":
            continue
            
        py_file = mutants_dir / f"{mid}.py"
        if not py_file.exists():
            continue
            
        mutant_analyzer = load_analyzer(py_file)
        result = run_tests_for_ms(original_analyzer, mutant_analyzer, 
                                  tests, include_detail=True)
        
        # 统计
        ms_vec = result['ms'].reshape(n_tests, 2)
        detail_vec = result['detail'].reshape(n_tests, 7)
        
        kill_count = np.sum(ms_vec[:, 1])
        
        # 细粒度分布
        detail_counts = np.sum(detail_vec, axis=0)
        detail_dist = {DETAIL_LABELS[i]: int(detail_counts[i]) 
                      for i in range(7)}
        
        analysis[mid] = {
            'operator': MUTANT_OPERATOR_MAP[mid],
            'killed': int(kill_count),
            'total': n_tests,
            'kill_rate': float(kill_count / n_tests),
            'detail_distribution': detail_dist
        }
    
    return analysis


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def cluster_by_coverage_only(coverage_vectors: Dict[str, np.ndarray], 
                              n_clusters: int = 9):
    """
    纯覆盖向量聚类（无AST融合）
    """
    # 提取变异体ID和向量矩阵
    mids = list(coverage_vectors.keys())
    # 排除M00如果存在
    mids = [m for m in mids if m != "M00"]
    
    # 构建特征矩阵: [n_mutants, n_tests * 2]
    # 或者压缩为: [n_mutants, n_tests]（只取"杀死"维度）
    X = np.array([coverage_vectors[m].reshape(-1, 2)[:, 1]  # 只取"杀死"概率
                  for m in mids])
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # 构建cluster_dict
    cluster_dict = {}
    for i, mid in enumerate(mids):
        cluster_id = f"cluster_{labels[i]}"
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(mid)
    
    # 计算轮廓系数评估聚类质量
    sil_score = silhouette_score(X, labels) if len(mids) > n_clusters else 0
    
    return cluster_dict, sil_score


# 选代表（每个聚类选杀死率最高的）
def select_representative_cov(cluster_members, coverage_vectors):
    best_mid = None
    best_kill_rate = -1
    for mid in cluster_members:
        vec = coverage_vectors[mid].reshape(-1, 2)
        kill_rate = np.mean(vec[:, 1])
        if kill_rate > best_kill_rate:
            best_kill_rate = kill_rate
            best_mid = mid
    return best_mid



# ============================================
# 8. 主程序入口（用于独立测试）
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("变异测试框架 - 正确MS算法实现")
    print("=" * 60)
    
    # 测试1：生成行为向量
    print("\n[1] 生成行为向量...")
    coverage_vectors = generate_coverage_vectors(n_tests=1000, seed=42)
    print(f"生成了 {len(coverage_vectors)} 个变异体的行为向量")
    
    # 验证M00（应该全部存活）
    if "M00" in coverage_vectors:
        m00_vec = coverage_vectors["M00"].reshape(-1, 2)
        m00_killed = np.sum(m00_vec[:, 1])
        print(f"验证M00（原程序）：杀死数 = {m00_killed}（应为0）")
    
    # 测试2：完整MS
    print("\n[2] 计算完整Mutation Score...")
    full_ms = compute_mutation_score(coverage_vectors)
    print(f"Full Mutation Score: {full_ms}")
    
    # 测试3：Test-averaged MS
    print("\n[3] 计算Test-averaged MS...")
    test_ms = compute_test_averaged_ms(coverage_vectors)
    print(f"Test-averaged MS: {test_ms}")
    
    # 测试4：细粒度分析（样本）
    print("\n[4] 细粒度分析（样本）...")
    detailed = generate_detailed_analysis(n_tests=100, seed=42)
    sample_mid = list(detailed.keys())[0]
    print(f"示例 {sample_mid}: {detailed[sample_mid]}")