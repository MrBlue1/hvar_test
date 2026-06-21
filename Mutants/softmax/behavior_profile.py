import os
import sys
import importlib.util
import numpy as np
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import entropy, qmc
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans

# ============================================
# 1. 配置和常量（保留你的原有配置）
# ============================================
mutants_dir = Path(__file__).parent

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
}

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

# MS算法用：二分类标签 [存活, 杀死]
MS_LABELS = ["存活", "杀死"]
ms_label2idx = {l: i for i, l in enumerate(MS_LABELS)}

# 【关键】7类细粒度行为指纹（用于行为约束聚类）
DETAIL_LABELS = [
    "正确", "数值溢出", "归一化错误", "负值错误",
    "形状错误", "类型错误", "其他异常"
]
detail_label2idx = {l: i for i, l in enumerate(DETAIL_LABELS)}

EPS = 1e-6
SUM_TOL = 1e-5

# ============================================
# 2. 测试数据生成（保留你的原有逻辑）
# ============================================
def lhs_samples(n: int, seed: Optional[int] = None, n_classes: int = 5, batch_size: int = 3):
    """生成 LHS 测试用例"""
    if seed:
        np.random.seed(seed)
    
    test_cases = []
    n_normal = int(n * 0.4)
    n_large = int(n * 0.3)
    n_small = n - n_normal - n_large
    
    def generate_layer(count, range_low, range_high):
        samples = []
        dims = batch_size * n_classes
        strata = np.linspace(0, 1, count, endpoint=False)
        
        dimensions = []
        for i in range(dims):
            jitter = np.random.uniform(0, 1/count, count)
            dim_samples = strata + jitter
            np.random.shuffle(dim_samples)
            dimensions.append(dim_samples * (range_high - range_low) + range_low)
        
        for i in range(count):
            logits = np.array([d[i] for d in dimensions]).reshape(batch_size, n_classes)
            samples.append(logits)
        return samples
    
    normal_samples = generate_layer(n_normal, -10, 10)
    large_samples = generate_layer(n_large, 100, 1000)
    small_samples = generate_layer(n_small, -1000, -100)
    
    all_logits = normal_samples + large_samples + small_samples
    np.random.shuffle(all_logits)
    
    temps = np.random.uniform(0.1, 5.0, n)
    axes = np.random.choice([-1, 0], n)
    
    for i in range(n):
        test_cases.append((all_logits[i], temps[i], axes[i]))
    
    # 插入边界用例
    boundary_cases = [
        (np.zeros((batch_size, n_classes)), 1.0, -1),
        (np.ones((batch_size, n_classes)) * 1000, 0.1, -1),
        (np.array([[0, 100, 0], [100, 0, 0], [0, 0, 100]]), 1.0, -1),
        (np.full((batch_size, n_classes), -1000), 1.0, -1),
    ]
    
    insert_indices = np.linspace(0, n-1, len(boundary_cases), dtype=int)
    for idx, case in zip(insert_indices, boundary_cases):
        test_cases[idx] = case
    
    return test_cases

# ============================================
# 3. 参考Oracle和分类（保留你的原有逻辑）
# ============================================
def reference_oracle(logits: np.ndarray, temperature: float = 1.0, axis: int = -1):
    """参考实现"""
    try:
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        scaled = shifted / temperature
        exp_x = np.exp(scaled)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        probs = exp_x / sum_exp
        probs = np.clip(probs, 1e-15, 1.0)
        return probs
    except Exception as e:
        raise RuntimeError(f"Oracle failed: {e}")

def check_numerical_issues(arr):
    """检查数组中的数值问题"""
    if arr is None:
        return False, False
    has_inf = np.any(np.isinf(arr))
    has_nan = np.any(np.isnan(arr))
    return has_inf, has_nan

def classify_detailed_behavior(expected: Optional[np.ndarray], actual, exception: Optional[Exception] = None, input_shape: Optional[Tuple] = None) -> str:
    """
    【关键】细粒度行为分类 - 用于行为指纹
    返回7类错误标签之一
    """
    if exception is not None:
        err_msg = str(exception).lower()
        if "overflow" in err_msg or "invalid value" in err_msg:
            return "数值溢出"
        elif isinstance(exception, (ZeroDivisionError, FloatingPointError)):
            return "数值溢出"
        elif isinstance(exception, TypeError):
            return "类型错误"
        elif isinstance(exception, (IndexError, ValueError)):
            return "形状错误"
        else:
            return "其他异常"
    
    if not isinstance(actual, np.ndarray):
        return "类型错误"
    
    has_inf, has_nan = check_numerical_issues(actual)
    if has_inf or has_nan:
        return "数值溢出"
    
    if input_shape is not None:
        expected_shape = input_shape
        if actual.shape != expected_shape:
            if len(actual.shape) != len(expected_shape):
                return "形状错误"
            for a, e in zip(actual.shape, expected_shape):
                if a != e and a != 1:
                    return "形状错误"
    
    if np.any(actual < -EPS):
        return "负值错误"
    
    try:
        sum_probs = np.sum(actual, axis=-1)
        if not np.all(np.abs(sum_probs - 1.0) < SUM_TOL):
            if np.any(sum_probs == 0) or np.any(np.isnan(sum_probs)):
                return "数值溢出"
            return "归一化错误"
    except Exception:
        return "形状错误"
    
    if expected is not None:
        try:
            if expected.shape != actual.shape:
                return "形状错误"
            if np.any(np.isnan(expected)) or np.any(np.isinf(expected)):
                return "数值溢出"
            
            max_abs = np.maximum(np.abs(expected), np.abs(actual))
            diff = np.abs(expected - actual)
            mask = max_abs > 1e-15
            relative_error = np.zeros_like(diff)
            np.divide(diff, max_abs, out=relative_error, where=mask)
            
            if np.any(relative_error > 1e-4):
                return "数值溢出"
        except Exception:
            return "其他异常"
    
    return "正确"

# ============================================
# 4. 核心：加载程序和运行测试（修改后）
# ============================================
def load_softmax_func(py_file: Path):
    """加载softmax函数"""
    name = py_file.stem
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.stable_softmax

def run_softmax_safe(softmax_func, logits: np.ndarray, temperature: float, axis: int):
    """安全运行softmax"""
    try:
        with np.errstate(all='raise'):
            result = softmax_func(logits, temperature=temperature, axis=axis)
        return result, None
    except Exception as e:
        return None, e

def outputs_equal(out1, out2, tol: float = 1e-6) -> bool:
    """比较两个输出是否相等（MS算法核心）"""
    if isinstance(out1, Exception) and isinstance(out2, Exception):
        return type(out1) == type(out2)
    
    if (out1 is None) != (out2 is None):
        return False
    if isinstance(out1, Exception) != isinstance(out2, Exception):
        return False
    
    if isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray):
        if out1.shape != out2.shape:
            return False
        for o1, o2 in [(out1, out2)]:
            if np.any(np.isnan(o1)) != np.any(np.isnan(o2)):
                return False
            if np.any(np.isinf(o1)) != np.any(np.isinf(o2)):
                return False
        
        mask = ~(np.isnan(out1) | np.isnan(out2))
        if np.any(mask):
            diff = np.abs(out1[mask] - out2[mask])
            max_abs = np.maximum(np.abs(out1[mask]), np.abs(out2[mask]))
            max_abs[max_abs < 1e-15] = 1
            relative_error = diff / max_abs
            return np.all(relative_error < tol)
        return True
    
    return out1 == out2

# ============================================
# 5. 【修改】生成覆盖率向量（同时生成MS向量和行为指纹）
# ============================================
def generate_coverage_vectors(n_tests: int = 2000, mutants_dir: Optional[Path] = None, seed: Optional[int] = 42,external_tests: Optional[List] = None) -> Dict[str, Dict]:
    """
    【关键修改】生成两类向量：
    1. ms_vector: [n_tests * 2] - 用于杀死矩阵（对比M00）
    2. behavior_vector: [n_tests * 7] - 用于行为指纹（细粒度分类）
    """
    if mutants_dir is None:
        mutants_dir = Path(__file__).parent
    
    # 【关键修改】如果提供了外部测试用例，直接使用；否则生成
    if external_tests is not None:
        tests = external_tests
        print(f"使用外部提供的 {len(tests)} 个测试用例（质量降级后）")
    else:
        print(f"生成 {n_tests} 个 LHS 测试用例...")
        tests = lhs_samples(n_tests, seed=seed)
        print(f"测试用例生成完成\n")
    
    # 加载原程序 M00.py
    original_path = mutants_dir / "M00.py"
    if not original_path.exists():
        raise FileNotFoundError(f"原程序不存在: {original_path}")
    
    original_func = load_softmax_func(original_path)    
    
    coverage_data = {}  # 存储两个字典
    total = len([m for m in MUTANT_OPERATOR_MAP if m != "M00"])
    mutant_ids = [m for m in MUTANT_OPERATOR_MAP if m != "M00"]
    
    for i, mid in enumerate(mutant_ids, 1):
        py_file = mutants_dir / f"{mid}.py"
        
        if not py_file.exists():
            print(f"[{i}/{total}] 警告: 未找到 {mid}.py，跳过")
            continue
            
        try:
            # print(f"[{i}/{total}] 测试 {mid} ({MUTANT_OPERATOR_MAP[mid]})...", end=' ')
            
            mutant_func = load_softmax_func(py_file)
            
            ms_results = []
            behavior_results = []
            
            for idx, (logits, temp, axis) in enumerate(tests):
                input_shape = logits.shape
                
                # 1. 运行原程序
                orig_out, orig_err = run_softmax_safe(original_func, logits, temp, axis)
                if orig_err:
                    orig_out = orig_err
                
                # 2. 运行变异体
                mut_out, mut_err = run_softmax_safe(mutant_func, logits, temp, axis)
                if mut_err:
                    mut_out = mut_err
                
                # 3. MS判断（杀死矩阵）
                is_killed = not outputs_equal(orig_out, mut_out)
                v_ms = np.zeros(2)
                v_ms[ms_label2idx["杀死" if is_killed else "存活"]] = 1
                ms_results.append(v_ms)
                
                # 4. 【关键】细粒度行为分类（行为指纹）
                expected = reference_oracle(logits, temp, axis) if not isinstance(orig_out, Exception) else None
                detail_label = classify_detailed_behavior(expected, mut_out, mut_err, input_shape)
                v_detail = np.zeros(7)
                v_detail[detail_label2idx[detail_label]] = 1
                behavior_results.append(v_detail)
            
            coverage_data[mid] = {
                'ms': np.concatenate(ms_results),           # [n_tests * 2]
                'behavior': np.concatenate(behavior_results) # [n_tests * 7]
            }
            
            # 快速统计
            ms_vec = coverage_data[mid]['ms'].reshape(n_tests, 2)
            kill_rate = np.mean(ms_vec[:, 1])
            # print(f"完成 | 杀死率: {kill_rate:.1%}")
                
        except Exception as e:
            print(f"错误: {e}")
            # Fallback
            fallback_ms = np.zeros(n_tests * 2)
            fallback_ms[0::2] = 1
            fallback_beh = np.zeros(n_tests * 7)
            fallback_beh[0::7] = 1  # 全部标记为"正确"
            coverage_data[mid] = {'ms': fallback_ms, 'behavior': fallback_beh}
    
    print(f"\n所有测试完成，共处理 {len(coverage_data)} 个变异体")
    return coverage_data

# ============================================
# 6. 【新增】数据解析与诊断模块
# ============================================
def extract_kill_matrix_and_profiles(coverage_data: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从coverage_data提取：
    - kill_matrix: (n_mutants, n_tests) 基于MS对比M00
    - behavior_profiles: (n_mutants, 7) 基于细粒度分类分布
    """
    mutant_ids = sorted(coverage_data.keys())
    n_mutants = len(mutant_ids)
    
    # 从第一个样本推断n_tests
    first_ms = coverage_data[mutant_ids[0]]['ms']
    n_tests = first_ms.size // 2
    
    kill_matrix = np.zeros((n_mutants, n_tests), dtype=int)
    behavior_profiles = np.zeros((n_mutants, 7))  # 7类细粒度行为
    
    for i, mid in enumerate(mutant_ids):
        data = coverage_data[mid]
        
        # 杀死矩阵：从MS向量提取（杀死=1，存活=0）
        ms_vec = data['ms'].reshape(n_tests, 2)
        kill_matrix[i] = ms_vec[:, 1]  # 杀死列
        
        # 行为指纹：从细粒度向量提取（每类出现频率）
        beh_vec = data['behavior'].reshape(n_tests, 7)
        behavior_profiles[i] = beh_vec.mean(axis=0)  # 统计分布
    
    return kill_matrix, behavior_profiles, mutant_ids

def five_minute_diagnosis(kill_matrix: np.ndarray, behavior_profiles: np.ndarray) -> Dict:
    """5分钟诊断（适配Softmax的7类行为）"""
    n_mutants, n_tests = kill_matrix.shape
    n_labels = 7  # Softmax有7类细粒度行为
    
    print(f"\n{'='*60}")
    print(f"【诊断报告】样本: {n_mutants}变异体 × {n_tests}测试用例")
    print(f"行为类别数: {n_labels}")
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
        risk_a, msg_a = "HIGH", "测试集过强，指纹增益有限"
    elif avg_jaccard < 0.3:
        risk_a, msg_a = "MEDIUM", "区分度较好"
    else:
        risk_a, msg_a = "LOW", "存在混淆，需要指纹辅助"
    print(f"  平均Jaccard: {avg_jaccard:.3f} - {risk_a}: {msg_a}")

    # 诊断B：模态相关性
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

    # 诊断C：行为多样性（7类）
    print("\n【诊断C】细粒度行为多样性")
    error_counts = np.bincount(dominant_errors, minlength=n_labels)
    unique_errors = np.sum(error_counts > 0)
    probs = error_counts / np.sum(error_counts)
    error_entropy = entropy(probs, base=2) / np.log2(n_labels) if n_labels > 1 else 0
    
    print(f"  出现类别: {unique_errors}/{n_labels}, 熵: {error_entropy:.3f}")
    print(f"  分布: {dict(zip(DETAIL_LABELS, error_counts))}")
    
    if unique_errors <= 3 or error_entropy < 0.4:
        risk_c = "HIGH"
    elif unique_errors <= 5:
        risk_c = "MEDIUM"
    else:
        risk_c = "LOW"
    print(f"  风险: {risk_c}")

    # 综合决策
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
def cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, k, force_one_per_class=True):
    """
    【算子优先修正版】严格确保FTRR不下降，且严格遵守预算k
    """
    n_mutants = len(mutant_ids)
    
    # 【严格】只考虑被杀死的变异体（排除等价变异体）
    killed_indices = []
    killed_mids = []
    for i, mid in enumerate(mutant_ids):
        if np.any(kill_matrix[i] == 1):  # 被杀死
            killed_indices.append(i)
            killed_mids.append(mid)
    
    if not killed_indices:
        print("警告：没有变异体被杀死，返回空结果")
        return {}
    
    # 在杀死的变异体中建立映射
    killed_kill_matrix = kill_matrix[killed_indices]
    killed_behavior_profiles = behavior_profiles[killed_indices]
    
    print(f"\n【算子优先聚类】总变异体:{n_mutants}, 被杀死的:{len(killed_indices)}, 预算k={k}")
    
    # 收集信息
    mutant_info = []
    for idx, mid in enumerate(killed_mids):
        i = killed_indices[idx]
        op_type = MUTANT_OPERATOR_MAP.get(mid, "UNKNOWN")
        beh_type = killed_behavior_profiles[idx].argmax()
        kill_count = killed_kill_matrix[idx].sum()
        
        mutant_info.append({
            'idx': i,
            'mid': mid,
            'op_type': op_type,
            'beh_type': beh_type,
            'kill_count': kill_count
        })
    
    # 按算子类型分组
    operator_groups = defaultdict(list)
    for m in mutant_info:
        operator_groups[m['op_type']].append(m)
    
    n_operator_types = len(operator_groups)
    print(f"算子类型数:{n_operator_types}类, 预算k={k}")
    
    # 确定目标算子类型
    if k >= n_operator_types:
        print(f"✓ 预算充足(k={k} >= 算子数{n_operator_types})，可覆盖所有算子类型")
        target_ops = list(operator_groups.keys())
    else:
        print(f"⚠️ 预算紧张(k={k} < 算子数{n_operator_types})，只覆盖{k}类")
        op_avg_kills = {
            op: np.mean([m['kill_count'] for m in members])
            for op, members in operator_groups.items()
        }
        target_ops = sorted(op_avg_kills.keys(), key=lambda x: -op_avg_kills[x])[:k]
        print(f"  选中算子类型: {target_ops}")
    
    representatives = {}
    selected_mids = set()
    cluster_id = 0
    
    # ==========================================
    # 阶段1：确保目标算子类型每类至少一个代表
    # 【修复】增加严格预算检查
    # ==========================================
    for op_type in target_ops:
        # 【关键修复】严格检查预算，超过则停止
        if len(representatives) >= k:
            print(f"\n⚠️ 预算已用尽({len(representatives)}/{k})，停止选择")
            break
            
        members = operator_groups[op_type]
        
        # 在该算子类型内，按行为分组
        beh_groups = defaultdict(list)
        for m in members:
            beh_groups[m['beh_type']].append(m)
        
        # 计算剩余预算
        remaining_budget = k - len(representatives)
        
        # 如果该算子有多样性行为，且预算允许，每类行为选一个代表
        if len(beh_groups) > 1 and remaining_budget > 1:
            # 最多选 min(行为类别数, 剩余预算, 3) 个，但至少留1个预算给后续算子（如果还有）
            # 如果这是最后一个算子，可以用完全部预算
            is_last_op = (op_type == target_ops[-1])
            if not is_last_op and len(target_ops) > 1:
                # 给后续算子预留：至少每个后续算子1个
                reserved = len(target_ops) - target_ops.index(op_type) - 1
                available = max(1, remaining_budget - reserved)
            else:
                available = remaining_budget
            
            n_select = min(len(beh_groups), available, 3)
            
            # 按行为分组，选杀死最多的
            sorted_beh = sorted(beh_groups.items(), key=lambda x: -len(x[1]))
            for i in range(n_select):
                # 【关键修复】每次选择前检查预算
                if len(representatives) >= k:
                    break
                    
                beh_type, beh_members = sorted_beh[i]
                best = max(beh_members, key=lambda x: x['kill_count'])
                beh_name = DETAIL_LABELS[beh_type]
                rep_name = f"C{cluster_id}_{op_type}_{beh_name}"
                representatives[rep_name] = [best['mid']]
                selected_mids.add(best['mid'])
                cluster_id += 1
                print(f"  选代表: {best['mid']} | 算子:{op_type} | 行为:{beh_name} | 杀死:{best['kill_count']}")
        else:
            # 只选该算子中杀死最多的1个
            # 【关键修复】检查预算
            if len(representatives) >= k:
                break
                
            best = max(members, key=lambda x: x['kill_count'])
            rep_name = f"C{cluster_id}_{op_type}"
            representatives[rep_name] = [best['mid']]
            selected_mids.add(best['mid'])
            cluster_id += 1
            print(f"  选代表: {best['mid']} | 算子:{op_type} | 杀死:{best['kill_count']}")
    
    # ==========================================
    # 阶段2：预算填充（严格受限）
    # ==========================================
    remaining_slots = k - len(representatives)
    
    # 【关键修复】确保remaining_slots不会为负数，且严格限制填充数量
    if remaining_slots > 0:
        remaining = [m for m in mutant_info if m['mid'] not in selected_mids]
        
        if remaining:
            remaining.sort(key=lambda x: -x['kill_count'])
            print(f"\n【预算填充】剩余{remaining_slots}个名额，补充高杀死率变异体:")
            
            # 【关键修复】使用切片严格限制数量：remaining[:remaining_slots]
            for m in remaining[:remaining_slots]:  
                if len(representatives) >= k:  # 双重保险
                    break
                    
                op_type = m['op_type']
                beh_name = DETAIL_LABELS[m['beh_type']]
                rep_name = f"C{cluster_id}_{op_type}_{beh_name}_sup"
                representatives[rep_name] = [m['mid']]
                selected_mids.add(m['mid'])
                cluster_id += 1
                print(f"  + {m['mid']} | 算子:{op_type} | 杀死:{m['kill_count']}")
    
    # 最终诊断
    covered_ops = set()
    for mids in representatives.values():
        for mid in mids:
            op = MUTANT_OPERATOR_MAP.get(mid, "UNKNOWN")
            covered_ops.add(op)
    
    print(f"\n【最终结果】选中{len(representatives)}个代表（预算k={k}）")
    print(f"  覆盖算子类型: {len(covered_ops)}/{n_operator_types} - {sorted(covered_ops)}")
    
    if len(representatives) > k:
        print(f"  ❌ 警告：超出预算！选中{len(representatives)} > k={k}")
    elif len(representatives) < k:
        print(f"  ⚠️  未用完预算：选中{len(representatives)} < k={k}（可能变异体不足或约束过严）")
    else:
        print(f"  ✅ 预算使用正好：{len(representatives)} = k={k}")
    
    return representatives


def baseline_cluster(kill_matrix, mutant_ids, k):
    """
    【分层 K-Means】对照组：先按算子类型分组，组内再做 K-Means
    确保与实验组公平对比（都使用 MUTANT_OPERATOR_MAP 信息）
    【修复】处理 distinct clusters 少于 n_clusters 的情况
    """
    from collections import defaultdict
    
    n_mutants = len(mutant_ids)
    
    # 【关键】同样使用算子信息分组（公平对比）
    operator_groups = defaultdict(list)
    for i, mid in enumerate(mutant_ids):
        op_type = MUTANT_OPERATOR_MAP.get(mid, "UNKNOWN")
        operator_groups[op_type].append(i)
    
    n_operators = len(operator_groups)
    print(f"\n【对照组-分层KMeans】算子类型数:{n_operators}, 预算k={k}")
    
    # 预算分配策略：每类算子至少1个，剩余按组大小比例分配
    base_quota = 1  # 每类至少1个
    remaining = k - n_operators * base_quota
    
    if remaining < 0:
        # 预算不足：只能覆盖 k 类算子（按组大小排序，大组优先）
        sorted_ops = sorted(operator_groups.items(), 
                           key=lambda x: -len(x[1]))  # 按组大小降序
        quotas = {op: 0 for op in operator_groups}
        for i, (op, _) in enumerate(sorted_ops[:k]):
            quotas[op] = 1
        print(f"⚠️ 预算不足(k={k} < 算子数{n_operators})，只覆盖{k}类大组")
    else:
        # 预算充足：每类1个 + 剩余按组大小分配
        quotas = {}
        total_members = sum(len(members) for members in operator_groups.values())
        for op, members in operator_groups.items():
            # 基础1个 + 按比例分配剩余
            extra = int(remaining * len(members) / total_members)
            quotas[op] = base_quota + extra
        
        # 处理剩余未分配的（由于取整误差）
        current_total = sum(quotas.values())
        if current_total < k:
            # 给最大的组再分配1个
            largest_op = max(operator_groups.keys(), key=lambda x: len(operator_groups[x]))
            quotas[largest_op] += (k - current_total)
    
    print(f"  预算分配: { {op: quotas[op] for op in sorted(quotas.keys())} }")
    
    representatives = {}
    cluster_id = 0
    
    # 对每个算子组分别聚类
    for op_type, indices in sorted(operator_groups.items()):
        quota = quotas.get(op_type, 0)
        if quota == 0:
            continue
            
        group_size = len(indices)
        
        if group_size <= quota:
            # 该组变异体数量 ≤ 配额，全部选中
            for idx in indices:
                mid = mutant_ids[idx]
                representatives[f"Base_{op_type}_{cluster_id}"] = [mid]
                cluster_id += 1
            print(f"  {op_type}: {group_size}个变异体 < 配额{quota}，全选")
        else:
            # 该组变异体数量 > 配额，进行 K-Means
            sub_matrix = kill_matrix[indices]
            
            # 【关键修复】检查实际的 distinct patterns 数量
            # 将每行转换为 tuple 以便去重计数
            unique_patterns = len(set(map(tuple, sub_matrix)))
            actual_clusters = min(quota, unique_patterns)
            
            if actual_clusters < quota:
                print(f"  ⚠️  {op_type}: 实际只有{unique_patterns}种独特杀死模式，调整簇数为{actual_clusters}")
            
            # 如果只有1种模式，直接选杀死最多的，不进行 K-Means
            if actual_clusters == 1:
                kill_counts = sub_matrix.sum(axis=1)
                best_idx = indices[np.argmax(kill_counts)]
                best_mid = mutant_ids[best_idx]
                representatives[f"Base_{op_type}_C{cluster_id}"] = [best_mid]
                cluster_id += 1
                print(f"  {op_type}: {group_size}个变异体只有1种模式，选1个代表")
            else:
                # 进行 K-Means（使用调整后的簇数）
                kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(sub_matrix)
                
                # 从每个子簇中选杀死最多的代表
                for sub_id in range(actual_clusters):
                    mask = (labels == sub_id)
                    cluster_local_indices = [i for i, m in enumerate(mask) if m]
                    
                    if cluster_local_indices:
                        # 映射回原始索引
                        cluster_global_indices = [indices[i] for i in cluster_local_indices]
                        # 选杀死最多的
                        kill_counts = kill_matrix[cluster_global_indices].sum(axis=1)
                        best_local = np.argmax(kill_counts)
                        best_idx = cluster_global_indices[best_local]
                        best_mid = mutant_ids[best_idx]
                        
                        representatives[f"Base_{op_type}_C{cluster_id}"] = [best_mid]
                        cluster_id += 1
                
                print(f"  {op_type}: {group_size}个变异体分{actual_clusters}簇（原预算{quota}），选{actual_clusters}个代表")
    
    print(f"【对照组】最终选中 {len(representatives)} 个代表")
    return representatives


def evaluate_reduction(coverage_data, representatives, mutant_ids):
    """计算约简质量（基于算子类型的FTRR）"""
    n_original = len(mutant_ids)
    n_reduced = sum(len(mids) for mids in representatives.values())
    
    # 【辅助函数】判断变异体是否被杀死（排除等价变异体）
    def is_killed(mid):
        if mid not in coverage_data:
            return False
        ms_vec = coverage_data[mid]['ms'].reshape(-1, 2)
        return np.any(ms_vec[:, 1] == 1)  # 杀死列是否有1
    
    # 【辅助函数】获取变异体的主导行为（用于对比分析）
    def get_dominant_behavior(mid):
        if mid not in coverage_data:
            return None
        beh_vec = coverage_data[mid]['behavior'].reshape(-1, 7)
        return np.argmax(beh_vec.mean(axis=0))
    
    # ==========================================
    # 1. 基于算子类型的FTRR（主要指标）
    # ==========================================
    # 收集所有被杀死的变异体的算子类型
    killed_mutants = [m for m in mutant_ids if is_killed(m)]
    all_operator_types = set(MUTANT_OPERATOR_MAP[m] for m in killed_mutants)
    
    # 收集代表变异体覆盖的算子类型（只考虑被杀死的代表）
    rep_operator_types = set()
    for cluster, mids in representatives.items():
        for mid in mids:
            if is_killed(mid):  # 确保代表是被杀死的
                op_type = MUTANT_OPERATOR_MAP.get(mid, "UNKNOWN")
                rep_operator_types.add(op_type)
    
    ftrr_operator = len(rep_operator_types) / len(all_operator_types) if all_operator_types else 0.0
    
    print(f"  [Debug] 原始算子类型: {sorted(all_operator_types)}")
    print(f"  [Debug] 代表算子类型: {sorted(rep_operator_types)}")
    print(f"  [Debug] FTRR计算: {len(rep_operator_types)}/{len(all_operator_types)}={ftrr_operator:.1%}")
    
    # ==========================================
    # 2. 基于行为指纹的FTRR（对比指标，可选）
    # ==========================================
    all_behavior_types = set(get_dominant_behavior(m) for m in killed_mutants)
    rep_behavior_types = set()
    for mids in representatives.values():
        for mid in mids:
            if is_killed(mid):
                rep_behavior_types.add(get_dominant_behavior(mid))
    
    ftrr_behavior = len(rep_behavior_types) / len(all_behavior_types) if all_behavior_types else 0.0
    
    # ==========================================
    # 3. MS保持率计算（保持不变）
    # ==========================================
    killed_original_set = set(killed_mutants)
    n_killed_original = len(killed_original_set)
    original_ms = n_killed_original / len(mutant_ids)
    
    rep_set = set()
    for mids in representatives.values():
        rep_set.update(mids)
    
    killed_retained_set = killed_original_set & rep_set
    n_killed_retained = len(killed_retained_set)
    ms_retention = n_killed_retained / n_killed_original if n_killed_original else 0.0
    
    # 代表纯度（代表中被杀死的比例）
    killed_in_rep = sum(1 for m in rep_set if is_killed(m))
    rep_purity = killed_in_rep / len(rep_set) if rep_set else 0.0
    
    return {
        'original_ms': original_ms,
        'reduced_ms': ms_retention,
        'rep_purity': rep_purity,
        'n_killed_original': n_killed_original,
        'n_killed_retained': n_killed_retained,
        'reduction_ratio': 1 - n_reduced/n_original,
        
        # 【关键】基于算子类型的FTRR（主要指标）
        'ftrr_operator': ftrr_operator,
        'all_operator_types': sorted(all_operator_types),
        'rep_operator_types': sorted(rep_operator_types),
        
        # 【对比】基于行为指纹的FTRR（辅助分析）
        'ftrr_behavior': ftrr_behavior,
        'all_behavior_types': [DETAIL_LABELS[i] for i in sorted(all_behavior_types)],
        'rep_behavior_types': [DETAIL_LABELS[i] for i in sorted(rep_behavior_types)],
        
        'representatives': representatives,
        'killed_retained_ids': list(killed_retained_set),
        'equivalent_count': len(mutant_ids) - n_killed_original
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


#region 绘制降维后聚类对比图形
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

# 中文到英文的映射（用于图表显示）
def set_chinese_font():
    """设置中文字体（Windows/Linux/Mac兼容）"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        # Windows常见中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong']
    elif system == 'Darwin':  # Mac
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
set_chinese_font()
def visualize_clusters(kill_matrix: np.ndarray, 
                      behavior_profiles: np.ndarray,
                      mutant_ids: List[str],
                      representatives: Dict[str, List[str]],
                      title: str = "聚类可视化",
                      method: str = "pca",
                      save_path: Optional[str] = None):
    """
    可视化聚类结果
    method: 'pca' 或 'tsne' (t-SNE较慢但效果更好)
    """
    n_mutants = len(mutant_ids)
    dominant_errors = behavior_profiles.argmax(axis=1)
    
    # 1. 降维到2D
    print(f"\n[{title}] 降维中...")
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(kill_matrix)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"PCA (解释方差: {explained_var[0]:.1%}, {explained_var[1]:.1%})"
    else:
        # t-SNE适合非线性结构，但较慢
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_mutants-1))
        coords = reducer.fit_transform(kill_matrix)
        subtitle = "t-SNE"
    
    # 2. 准备颜色映射（基于行为类型）
    unique_behaviors = np.unique(dominant_errors)
    colors = plt.cm.tab10(np.linspace(0, 1, len(DETAIL_LABELS)))
    
    # 3. 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 4. 绘制所有变异体（按行为类型着色，透明度0.6）
    for i, behavior_idx in enumerate(unique_behaviors):
        mask = (dominant_errors == behavior_idx)
        label_name = DETAIL_LABELS[behavior_idx]
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                  c=[colors[behavior_idx]], 
                  label=label_name,
                  alpha=0.6, s=50, edgecolors='none')
    
    # 5. 标记选中的代表（黑色五角星，大尺寸）
    rep_indices = []
    rep_labels = []  # 记录代表属于哪个簇
    for cluster_name, mids in representatives.items():
        for mid in mids:
            if mid in mutant_ids:
                idx = mutant_ids.index(mid)
                rep_indices.append(idx)
                rep_labels.append(cluster_name)
    
    if rep_indices:
        ax.scatter(coords[rep_indices, 0], coords[rep_indices, 1],
                  c='black', marker='*', s=300, 
                  edgecolors='white', linewidths=1.5,
                  label=f'代表 (n={len(rep_indices)})', zorder=5)
        
        # 可选：添加代表ID标签（如果数量少）
        if len(rep_indices) <= 15:
            for idx, label in zip(rep_indices, rep_labels):
                ax.annotate(mutant_ids[idx], 
                           (coords[idx, 0], coords[idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    # 6. 美化图表
    ax.set_xlabel(f'Component 1', fontsize=12)
    ax.set_ylabel(f'Component 2', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, pad=20)
    
    # 7. 双列图例（行为类型 + 代表标记）
    legend1 = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                       title="行为类型", frameon=True)
    ax.add_artist(legend1)
    
    # 添加统计信息文本框
    stats_text = f"变异体总数: {n_mutants}\n"
    stats_text += f"选中代表数: {len(rep_indices)}\n"
    stats_text += f"行为类别数: {len(unique_behaviors)}\n"
    stats_text += f"FTRR: {len(set(dominant_errors[rep_indices]))}/{len(unique_behaviors)}"
    
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def compare_visualization(kill_matrix, behavior_profiles, mutant_ids,
                         reps_base, reps_exp, method="pca"):
    """
    并排对比两组聚类结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    dominant_errors = behavior_profiles.argmax(axis=1)
    unique_behaviors = np.unique(dominant_errors)
    colors = plt.cm.tab10(np.linspace(0, 1, len(DETAIL_LABELS)))
    
    # 降维（统一坐标系，确保可比性）
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(kill_matrix)
    else:
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(mutant_ids)-1)).fit_transform(kill_matrix)
    
    for ax_idx, (reps, title) in enumerate([(reps_base, "对照组 (K-Means)"), 
                                            (reps_exp, "实验组 (行为约束)")]):
        ax = axes[ax_idx]
        
        # 绘制所有点
        for i, behavior_idx in enumerate(unique_behaviors):
            mask = (dominant_errors == behavior_idx)
            ax.scatter(coords[mask, 0], coords[mask, 1], 
                      c=[colors[behavior_idx]], 
                      label=DETAIL_LABELS[behavior_idx] if ax_idx == 0 else "",
                      alpha=0.5, s=50)
        
        # 标记代表
        rep_indices = [mutant_ids.index(m) for mids in reps.values() for m in mids if m in mutant_ids]
        if rep_indices:
            ax.scatter(coords[rep_indices, 0], coords[rep_indices, 1],
                      c='red', marker='*', s=400, 
                      edgecolors='black', linewidths=2,
                      label='代表', zorder=5)
        
        # 计算FTRR显示
        rep_behaviors = set(dominant_errors[rep_indices])
        ftrr = len(rep_behaviors) / len(unique_behaviors)
        
        ax.set_title(f"{title}\nFTRR: {ftrr:.1%} ({len(rep_behaviors)}/{len(unique_behaviors)}类)", 
                    fontsize=14)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        
        # 添加网格
        ax.grid(True, alpha=0.3)
    
    # 共享图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=len(DETAIL_LABELS)//2 + 1, title="行为类型")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#endregion


def degrade_test_quality(test_cases: List[Tuple], removal_ratio: float = 0.2, 
                        strategy: str = "boundary") -> List[Tuple]:
    """
    人为降低测试质量，用于验证测试充分性下限
    
    Args:
        test_cases: 生成的测试用例列表 [(logits, temp, axis), ...]
        removal_ratio: 移除比例 (0.0-1.0)
        strategy: 移除策略
            - "boundary": 优先移除边界值（极端值）
            - "random": 随机移除
            - "central": 优先移除中心值（正常范围）
    
    Returns:
        降级后的测试用例列表
    """
    if not test_cases or removal_ratio <= 0:
        return test_cases
    
    n_remove = int(len(test_cases) * removal_ratio)
    if n_remove == 0:
        return test_cases
    
    # 计算每个测试用例的"边界程度"（数值范围的极端性）
    boundary_scores = []
    for logits, temp, axis in test_cases:
        # 使用数值的绝对值最大值作为边界程度指标
        max_val = np.max(np.abs(logits))
        # 温度值的极端性（越接近0或越大越极端）
        temp_extreme = max(1/temp, temp) if temp > 0 else 999
        
        # 综合边界分数：数值极端性 + 温度极端性
        score = max_val + temp_extreme * 10
        boundary_scores.append(score)
    
    # 根据策略选择要移除的索引
    indices = list(range(len(test_cases)))
    
    if strategy == "boundary":
        # 优先移除边界值（分数最高的）
        sorted_indices = np.argsort(boundary_scores)[::-1]  # 降序
        remove_indices = set(sorted_indices[:n_remove])
        print(f"  [降级] 移除了{n_remove}个边界测试用例（最大值{max(boundary_scores):.1f}）")
    elif strategy == "central":
        # 优先移除中心值（分数最低的）
        sorted_indices = np.argsort(boundary_scores)  # 升序
        remove_indices = set(sorted_indices[:n_remove])
        print(f"  [降级] 移除了{n_remove}个中心测试用例")
    else:  # random
        np.random.shuffle(indices)
        remove_indices = set(indices[:n_remove])
        print(f"  [降级] 随机移除了{n_remove}个测试用例")
    
    # 保留未被移除的
    remaining = [tc for i, tc in enumerate(test_cases) if i not in remove_indices]
    return remaining


# ============================================
# 8. 主流程（Softmax验证专用）
# ============================================
def main():
    # 配置
    N_TESTS = 500  # 从20开始测试（资源受限场景）
    SEED = 42
    
    print("=" * 60)
    print("Softmax 行为指纹验证实验")
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
    # 【修改】使用 ftrr_operator 和 rep_operator_types
    print(f"  FTRR: {eval_base['ftrr_operator']:.1%} ({len(eval_base['rep_operator_types'])}/{len(eval_base['all_operator_types'])}类)")
    print(f"  保留算子: {eval_base['rep_operator_types']}")
    
    # 实验组（现在传入k_auto并严格遵守）
    print(f"\n实验组 (行为约束聚类, k={k_auto}):")
    reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, 
                                                k=k_auto, force_one_per_class=True)
    eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
    print(f"  代表数: {len(reps_exp)}")
    # 【修改】使用 ftrr_operator 和 rep_operator_types
    print(f"  FTRR: {eval_exp['ftrr_operator']:.1%} ({len(eval_exp['rep_operator_types'])}/{len(eval_exp['all_operator_types'])}类)")
    print(f"  保留算子: {eval_exp['rep_operator_types']}")
    
    # 对比结果
    print(f"\n{'='*60}")
    print("【对比结果】")
    print(f"预算k值: {k_auto}")
    print(f"{'='*60}")


    # ============================================
    # 【新增】可视化部分
    # ============================================
    print(f"\n{'='*60}")
    print("【聚类结果可视化】")
    print(f"{'='*60}")
    
    # 方法1：分别可视化（适合详细查看）
    # visualize_clusters(kill_matrix, behavior_profiles, mutant_ids, 
    #                   reps_base, 
    #                   title="对照组：纯Kill Matrix K-Means",
    #                   method="pca",
    #                   save_path="baseline_cluster.png")
    
    # visualize_clusters(kill_matrix, behavior_profiles, mutant_ids, 
    #                   reps_exp, 
    #                   title="实验组：行为约束聚类",
    #                   method="pca", 
    #                   save_path="behavior_constraint_cluster.png")
    
    # # 方法2：并排对比（适合论文插图）
    # compare_visualization(kill_matrix, behavior_profiles, mutant_ids,
    #                      reps_base, reps_exp, method="pca")

    
    # 对比结果输出（详细版）
    print(f"\n{'='*60}")
    print("【对比结果 - 基于算子类型的FTRR】")
    print(f"{'='*60}")
    
    # FTRR 对比（基于算子类型）
    print(f"\n【故障类型保留率 (FTRR - Operator Based)】")
    print(f"对照组: {eval_base['ftrr_operator']:.1%} ({len(eval_base['rep_operator_types'])}/{len(eval_base['all_operator_types'])}类)")
    print(f"  保留算子: {eval_base['rep_operator_types']}")
    print(f"  原始算子: {eval_base['all_operator_types']}")
    
    print(f"实验组: {eval_exp['ftrr_operator']:.1%} ({len(eval_exp['rep_operator_types'])}/{len(eval_exp['all_operator_types'])}类)")
    print(f"  保留算子: {eval_exp['rep_operator_types']}")
    
    # 【诊断】检查实验组是否丢失算子类型
    lost_operators = set(eval_base['all_operator_types']) - set(eval_exp['rep_operator_types'])
    if lost_operators:
        print(f"⚠️  实验组丢失算子类型: {sorted(lost_operators)}")
    else:
        print(f"✅ 实验组保留了所有算子类型")
    
    # 【对比】行为指纹FTRR（用于分析差异）
    print(f"\n【行为指纹保留率 (FTRR - Behavioral) - 对比参考】")
    print(f"对照组: {eval_base['ftrr_behavior']:.1%} ({len(eval_base['rep_behavior_types'])}/{len(eval_base['all_behavior_types'])}类)")
    print(f"实验组: {eval_exp['ftrr_behavior']:.1%} ({len(eval_exp['rep_behavior_types'])}/{len(eval_exp['all_behavior_types'])}类)")
    
    # 如果两者有差异，提示分析
    if abs(eval_exp['ftrr_operator'] - eval_exp['ftrr_behavior']) > 0.1:
        print(f"\n【差异分析】算子FTRR ({eval_exp['ftrr_operator']:.1%}) vs 行为FTRR ({eval_exp['ftrr_behavior']:.1%})")
        print("  说明：某些算子类型产生了相同的行为模式，导致行为聚类覆盖了更多算子")
    
    # MS 保持率（保持不变）
    print(f"\n{'='*60}")
    print("【变异分数保持率 (MS Retention)】")
    print(f"{'='*60}")
    print(f"原始全集杀死率: {eval_base['original_ms']:.1%} ({eval_base['n_killed_original']}/60个变异体被杀死)")
    print(f"等价变异体数: {eval_base['equivalent_count']}个 (未计入FTRR)")
    
    print(f"\n对照组 (K-Means):")
    print(f"  保留杀死数: {eval_base['n_killed_retained']}/{eval_base['n_killed_original']}")
    print(f"  MS 保持率: {eval_base['reduced_ms']:.1%}")
    print(f"  代表纯度: {eval_base['rep_purity']:.1%}")
    
    print(f"\n实验组 (行为约束):")
    print(f"  保留杀死数: {eval_exp['n_killed_retained']}/{eval_exp['n_killed_original']}")
    print(f"  MS 保持率: {eval_exp['reduced_ms']:.1%}")
    print(f"  代表纯度: {eval_exp['rep_purity']:.1%}")
    
    # MS 损失分析
    loss_base = 1 - eval_base['reduced_ms']
    loss_exp = 1 - eval_exp['reduced_ms']
    print(f"\n【MS 损失分析】")
    print(f"对照组损失: {loss_base:.1%} (丢弃 {eval_base['n_killed_original'] - eval_base['n_killed_retained']} 个杀死变异体)")
    print(f"实验组损失: {loss_exp:.1%} (丢弃 {eval_exp['n_killed_original'] - eval_exp['n_killed_retained']} 个杀死变异体)")
    

    # 对照组（使用相同的k_auto，且同样基于算子信息）
    print(f"\n对照组 (分层K-Means, 算子感知, k={k_auto}):")
    reps_base = baseline_cluster(kill_matrix, mutant_ids, k=k_auto)
    eval_base = evaluate_reduction(coverage_data, reps_base, mutant_ids)
    print(f"  代表数: {len(reps_base)}")
    print(f"  FTRR: {eval_base['ftrr_operator']:.1%} ({len(eval_base['rep_operator_types'])}/{len(eval_base['all_operator_types'])}类)")
    print(f"  保留算子: {eval_base['rep_operator_types']}")
    
    # 实验组
    print(f"\n实验组 (算子感知+行为约束, k={k_auto}):")
    reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, 
                                                k=k_auto, force_one_per_class=True)
    eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
    print(f"  代表数: {len(reps_exp)}")
    print(f"  FTRR: {eval_exp['ftrr_operator']:.1%} ({len(eval_exp['rep_operator_types'])}/{len(eval_exp['all_operator_types'])}类)")
    print(f"  保留算子: {eval_exp['rep_operator_types']}")

    # 综合评估
    print(f"\n{'='*60}")
    print("【综合评估】")
    print(f"{'='*60}")
    if eval_exp['ftrr_operator'] > eval_base['ftrr_operator']:
        print(f"✅ FTRR 提升: {eval_base['ftrr_operator']:.1%} → {eval_exp['ftrr_operator']:.1%}")
    elif eval_exp['ftrr_operator'] < eval_base['ftrr_operator']:
        print(f"⚠️  FTRR 下降: {eval_base['ftrr_operator']:.1%} → {eval_exp['ftrr_operator']:.1%}")
    else:
        print(f"➡️  FTRR 持平: {eval_exp['ftrr_operator']:.1%}")

def main1():
    N_TESTS = 200
    SEED = 42
    
    # 【修改】定义要测试的 K 值列表
    k_values = [3, 5, 8, 10, 15, 20, 25]
    
    # 生成数据（只需要生成一次，重复使用）
    print(f"生成 {N_TESTS} 个测试用例...")
    coverage_data = generate_coverage_vectors(n_tests=N_TESTS, seed=SEED)
    kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(coverage_data)
    
    # 存储结果用于后续分析/绘图
    results = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"【预算 K = {k}】")
        print(f"{'='*60}")
        
        # 对照组
        reps_base = baseline_cluster(kill_matrix, mutant_ids, k=k)
        eval_base = evaluate_reduction(coverage_data, reps_base, mutant_ids)
        
        # 实验组
        reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, 
                                                    k=k, force_one_per_class=True)
        eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
        
        # 记录结果
        result = {
            'k': k,
            'baseline_ms': eval_base['reduced_ms'],
            'baseline_ftrr': eval_base['ftrr_operator'],
            'proposed_ms': eval_exp['reduced_ms'],
            'proposed_ftrr': eval_exp['ftrr_operator'],
            'improvement': (eval_exp['reduced_ms'] - eval_base['reduced_ms']) * 100
        }
        results.append(result)
        
        # 打印对比
        print(f"\n结果对比:")
        print(f"  FTRR:  {eval_base['ftrr_operator']:.1%} vs {eval_exp['ftrr_operator']:.1%}")
        print(f"  MS:    {eval_base['reduced_ms']:.1%} vs {eval_exp['reduced_ms']:.1%} "
              f"(+{result['improvement']:.1f}%)")
    
    # 最终汇总表
    print(f"\n{'='*60}")
    print("【汇总表】不同预算下的 MS 保持率")
    print(f"{'='*60}")
    print(f"{'K':<5} {'对照组':<10} {'实验组':<10} {'提升':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['k']:<5} {r['baseline_ms']:<10.1%} {r['proposed_ms']:<10.1%} "
              f"+{r['improvement']:.1f}%")
    
    # 可选：保存到 CSV 或绘图
    # import pandas as pd
    # df = pd.DataFrame(results)
    # df.to_csv("k_comparison.csv", index=False)
    # print("\n结果已保存到 k_comparison.csv")

def main2():
    SEED = 42
    
    # 测试不同规模，固定比例（如保留 15%）
    for N_TESTS in [20, 50, 100, 200]:
        print(f"\n{'='*60}")
        print(f"【测试规模: {N_TESTS} 个测试用例】")
        print(f"{'='*60}")
        
        coverage_data = generate_coverage_vectors(n_tests=N_TESTS, seed=SEED)
        kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(coverage_data)
        
        # 【关键】根据 N_TESTS 动态调整 K（测试越多，预算越多）
        # 策略：K = min(算子类型数, int(N_TESTS * 0.05)) 
        # 即每 20 个测试用例允许 1 个代表
        k_dynamic = min(8, max(3, int(N_TESTS * 0.05)))
        print(f"动态预算 K = {k_dynamic} (基于 {N_TESTS} 个测试用例)")
        
        # 对照组
        reps_base = baseline_cluster(kill_matrix, mutant_ids, k=k_dynamic)
        eval_base = evaluate_reduction(coverage_data, reps_base, mutant_ids)
        
        # 实验组
        reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, k=k_dynamic)
        eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
        
        print(f"\n结果对比 (K={k_dynamic}):")
        print(f"  对照组 - FTRR: {eval_base['ftrr_operator']:.1%}, MS保持率: {eval_base['reduced_ms']:.1%}")
        print(f"  实验组 - FTRR: {eval_exp['ftrr_operator']:.1%}, MS保持率: {eval_exp['reduced_ms']:.1%}")
        print(f"  提升幅度: MS保持率 +{(eval_exp['reduced_ms']-eval_base['reduced_ms'])*100:.1f}%")

# 二维实验：测试用例规模 × 预算约束
#     生成完整的实验矩阵
def main4():
    """
    二维实验：测试用例规模 × 预算约束
    生成完整的实验矩阵
    """
    SEED = 42
    
    # 【实验配置】
    TEST_SIZES = [20, 50, 100, 200]  # 测试用例规模
    K_VALUES = [ 5, 10,  20, 25]  # 预算约束
    
    print("=" * 70)
    print("【二维实验】测试用例规模 × 预算约束")
    print("=" * 70)
    print(f"测试规模: {TEST_SIZES}")
    print(f"预算取值: {K_VALUES}")
    print("=" * 70)
    
    # 存储所有结果
    all_results = []
    
    # 外层循环：不同测试用例规模
    for n_tests in TEST_SIZES:
        print(f"\n{'='*70}")
        print(f"【测试用例规模 N = {n_tests}】")
        print(f"{'='*70}")
        
        # 生成数据（每个规模只生成一次，重复使用）
        coverage_data = generate_coverage_vectors(n_tests=n_tests, seed=SEED)
        kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(coverage_data)
        
        # 诊断信息
        n_killed = sum(1 for m in mutant_ids if np.any(coverage_data[m]['ms'].reshape(-1, 2)[:, 1] == 1))
        print(f"诊断: 被杀死变异体 = {n_killed}/60, Kill Matrix 形状 = {kill_matrix.shape}")
        
        # 内层循环：不同预算
        for k in K_VALUES:
            # 安全检查：k 不能超过被杀死的变异体数
            if k > n_killed:
                print(f"  [K={k}] 跳过（超过被杀死变异体数 {n_killed}）")
                continue
            
            print(f"\n  [K={k}] ", end="")
            
            # 对照组：分层 K-Means
            reps_base = baseline_cluster(kill_matrix, mutant_ids, k=k)
            eval_base = evaluate_reduction(coverage_data, reps_base, mutant_ids)
            
            # 实验组：行为约束聚类
            reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, 
                                                       k=k, force_one_per_class=True)
            eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
            
            # 记录结果
            result = {
                'n_tests': n_tests,
                'k': k,
                'n_killed': n_killed,
                'baseline_ms': eval_base['reduced_ms'],
                'baseline_ftrr': eval_base['ftrr_operator'],
                'proposed_ms': eval_exp['reduced_ms'],
                'proposed_ftrr': eval_exp['ftrr_operator'],
                'improvement_ms': (eval_exp['reduced_ms'] - eval_base['reduced_ms']) * 100,
                'improvement_ftrr': (eval_exp['ftrr_operator'] - eval_base['ftrr_operator']) * 100
            }
            all_results.append(result)
            
            print(f"MS: {eval_base['reduced_ms']:.1%} → {eval_exp['reduced_ms']:.1%} "
                  f"(+{result['improvement_ms']:.1f}%) | "
                  f"FTRR: {eval_base['ftrr_operator']:.1%} vs {eval_exp['ftrr_operator']:.1%}")
    
    # ============================================
    # 汇总输出
    # ============================================
    print(f"\n{'='*70}")
    print("【实验结果汇总】")
    print(f"{'='*70}")
    
    # 按测试规模分组打印表格
    for n_tests in TEST_SIZES:
        print(f"\n测试用例数 = {n_tests}:")
        print(f"{'K':<5} {'Baseline MS':<12} {'Proposed MS':<12} {'提升':<10} {'FTRR 提升':<10}")
        print("-" * 60)
        
        for r in all_results:
            if r['n_tests'] == n_tests:
                print(f"{r['k']:<5} "
                      f"{r['baseline_ms']:<12.1%} "
                      f"{r['proposed_ms']:<12.1%} "
                      f"+{r['improvement_ms']:<9.1f}% "
                      f"+{r['improvement_ftrr']:<9.1f}%")
    
    # ============================================
    # 保存为 CSV（可选）
    # ============================================
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_filename = f"experiment_matrix_seed{SEED}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n结果已保存到: {csv_filename}")
        
        # 绘制热力图（如果安装了 seaborn）
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # 创建透视表：行为 K，列为 N_TESTS，值为 MS 提升
            pivot_improvement = df.pivot(index='k', columns='n_tests', values='improvement_ms')
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_improvement, annot=True, fmt=".1f", cmap="YlGnBu", 
                       cbar_kws={'label': 'MS Improvement (%)'})
            plt.title('MS Retention Improvement (%) across Test Size and Budget')
            plt.xlabel('Number of Test Cases (N)')
            plt.ylabel('Budget (K)')
            plt.tight_layout()
            plt.savefig("improvement_heatmap.png", dpi=300)
            print("热力图已保存到: improvement_heatmap.png")
            plt.show()
            
        except ImportError:
            print("提示: 安装 seaborn 和 matplotlib 可绘制热力图")
            
    except ImportError:
        print("提示: 安装 pandas 可保存 CSV 结果")
    
    # ============================================
    # 关键发现分析
    # ============================================
    print(f"\n{'='*70}")
    print("【关键发现】")
    print(f"{'='*70}")
    
    # 找出最大提升
    max_imp = max(all_results, key=lambda x: x['improvement_ms'])
    print(f"最大 MS 提升: +{max_imp['improvement_ms']:.1f}% "
          f"(N={max_imp['n_tests']}, K={max_imp['k']})")
    
    # 找出最优性价比（K 小但提升大）
    good_configs = [r for r in all_results if r['improvement_ms'] > 20 and r['k'] <= 15]
    if good_configs:
        print(f"\n高性价比配置 (K≤15, 提升>20%):")
        for r in sorted(good_configs, key=lambda x: x['improvement_ms'], reverse=True)[:3]:
            print(f"  N={r['n_tests']}, K={r['k']}: +{r['improvement_ms']:.1f}%")
    
    # 分析测试充分性影响
    print(f"\n测试充分性影响分析 (固定 K=15):")
    k15_results = [r for r in all_results if r['k'] == 15]
    if len(k15_results) > 1:
        for r in sorted(k15_results, key=lambda x: x['n_tests']):
            print(f"  N={r['n_tests']:3d}: MS {r['baseline_ms']:.1%} → {r['proposed_ms']:.1%} "
                  f"(+{r['improvement_ms']:.1f}%)")

def main5():
    SEED = 42
    
    # 对比：原始 vs 移除30%边界值 vs 移除50%边界值
    configs = [
        ("原始质量", 20, 0.0),
        ("轻度降级", 20, 0.3),  # 移除30%边界值，剩余14个
        ("重度降级", 20, 0.5),  # 移除50%边界值，剩余10个
        ("原始质量", 100, 0.0),
        ("轻度降级", 100, 0.3),
    ]
    
    for label, n_tests, degrade_ratio in configs:
        print(f"\n{'='*60}")
        print(f"【{label}】N={n_tests}, 降级={degrade_ratio*100:.0f}%")
        print(f"{'='*60}")
        
        # 1. 生成原始测试
        raw_tests = lhs_samples(n_tests, seed=SEED)
        
        # 2. 应用降级
        if degrade_ratio > 0:
            final_tests = degrade_test_quality(raw_tests, degrade_ratio, strategy="boundary")
        else:
            final_tests = raw_tests
        
        # 3. 使用降级后的测试运行（传入 external_tests）
        coverage_data = generate_coverage_vectors(
            n_tests=len(final_tests),  # 传入数量用于数组初始化
            seed=SEED,
            external_tests=final_tests  # 【关键】传入降级后的测试
        )
        
        # 4. 后续聚类分析...
        kill_matrix, behavior_profiles, mutant_ids = extract_kill_matrix_and_profiles(coverage_data)
        
        # 固定 K=15 进行对比
        k = 15
        reps_base = baseline_cluster(kill_matrix, mutant_ids, k=k)
        eval_base = evaluate_reduction(coverage_data, reps_base, mutant_ids)
        
        reps_exp = cluster_with_behavior_constraint(kill_matrix, behavior_profiles, mutant_ids, k=k)
        eval_exp = evaluate_reduction(coverage_data, reps_exp, mutant_ids)
        
        print(f"结果: Baseline MS={eval_base['reduced_ms']:.1%}, "
              f"Proposed MS={eval_exp['reduced_ms']:.1%}, "
              f"杀死变异体数={eval_base['n_killed_original']}")

def diagnose_low_kill_mutants(mutants_dir: Path, coverage_data: Dict):
    """
    诊断杀死率为0的变异体：区分"真等价" vs "测试不充分"
    针对你的实验结果：ABS(M55-M60), LOR(M23-M30), ROR(M01-M02,M04-M12)
    """
    print(f"\n{'='*60}")
    print("【变异体存活原因诊断】")
    print(f"{'='*60}")
    
    # 从coverage_data找出存活变异体（杀死率0%）
    alive_mutants = []
    for mid, data in coverage_data.items():
        if mid == "M00":
            continue
        ms_vec = data['ms'].reshape(-1, 2)
        kill_rate = np.mean(ms_vec[:, 1])
        if kill_rate == 0.0:
            alive_mutants.append(mid)
    
    print(f"发现 {len(alive_mutants)} 个存活变异体: {alive_mutants}")
    
    # 加载原始程序
    original_func = load_softmax_func(mutants_dir / "M00.py")
    
    # 构造针对性测试用例（弥补LHS的缺陷）
    diagnostic_tests = {
        "负数输入": (np.array([[-1.0, -2.0, -3.0], 
                              [-10.0, -5.0, -1.0], 
                              [-100.0, -50.0, -1.0]]), 1.0, -1),
        "零输入": (np.zeros((3, 3)), 1.0, -1),
        "极大正数": (np.array([[1000.0, 2000.0, 3000.0],
                             [1e10, 1e9, 1e8],
                             [999999.0, 1000000.0, 1000001.0]]), 0.1, -1),
        "极大负数": (np.array([[-1000.0, -2000.0, -3000.0],
                             [-1e10, -1e9, -1e8]]), 0.1, -1),
        "混合符号": (np.array([[-5.0, 0.0, 5.0],
                             [-100.0, 1.0, 100.0]]), 1.0, -1),
        "温度极值": (np.array([[1.0, 2.0, 3.0]]), 0.001, -1),  # 接近0的温度
    }
    
    results = {}
    
    for mid in alive_mutants[:5]:  # 只检查前5个，避免输出过长
        print(f"\n检查 {mid} ({MUTANT_OPERATOR_MAP.get(mid, 'UNK')})...")
        mutant_func = load_softmax_func(mutants_dir / f"{mid}.py")
        
        killed_by = []
        equivalent = True
        
        for test_name, (logits, temp, axis) in diagnostic_tests.items():
            # 运行原始程序
            orig_out, orig_err = run_softmax_safe(original_func, logits, temp, axis)
            if orig_err:
                orig_out = orig_err
                
            # 运行变异体
            mut_out, mut_err = run_softmax_safe(mutant_func, logits, temp, axis)
            if mut_err:
                mut_out = mut_err
            
            # 检查是否被杀死（使用你的outputs_equal逻辑）
            if not outputs_equal(orig_out, mut_out):
                killed_by.append(test_name)
                equivalent = False
                print(f"  ✓ 被 '{test_name}' 杀死")
                print(f"    原始: {orig_out if not isinstance(orig_out, Exception) else str(orig_out)[:50]}")
                print(f"    变异: {mut_out if not isinstance(mut_out, Exception) else str(mut_out)[:50]}")
                break  # 找到一个能杀死的即可证明非等价
        
        if equivalent:
            print(f"  ✗ 所有测试下均等价 → 可能是真等价体")
        
        results[mid] = {
            "operator": MUTANT_OPERATOR_MAP.get(mid, "UNK"),
            "equivalent": equivalent,
            "killed_by": killed_by
        }
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("【诊断结论】")
    print(f"{'='*60}")
    
    by_op = defaultdict(lambda: {"total": 0, "equivalent": 0, "test_inadequate": 0})
    for mid, res in results.items():
        op = res["operator"]
        by_op[op]["total"] += 1
        if res["equivalent"]:
            by_op[op]["equivalent"] += 1
        else:
            by_op[op]["test_inadequate"] += 1
    
    for op, stats in by_op.items():
        print(f"{op}: 检查{stats['total']}个")
        print(f"  真等价体: {stats['equivalent']}")
        print(f"  测试不充分: {stats['test_inadequate']}")
        if stats["test_inadequate"] > 0:
            print(f"  → 建议：补充特定测试用例（如负数输入）以杀死这类变异体")
    
    return results

def quick_check_abs_lor(mutants_dir: Path):
    """
    快速验证：ABS(M55)和LOR(M23)的具体行为对比
    直接复用你的现有函数
    """
    print(f"\n{'='*60}")
    print("【快速验证：ABS vs LOR 等价性】")
    print(f"{'='*60}")
    
    original_func = load_softmax_func(mutants_dir / "M00.py")
    
    # 选取代表：M55(ABS,存活), M23(LOR,存活)
    test_cases = [
        ("ABS变异体M55", "M55"),
        ("LOR变异体M23", "M23"),
    ]
    
    # 关键测试用例：负数输入（LHS缺失的场景）
    test_input = (np.array([[-5.0, -3.0, -1.0], 
                           [-100.0, 50.0, -50.0]]),  # 混合负数
                  1.0, -1)
    
    logits, temp, axis = test_input
    
    print(f"\n测试输入: 负数矩阵")
    print(f"Logits: {logits}")
    
    # 运行原始程序
    orig_out, _ = run_softmax_safe(original_func, logits, temp, axis)
    print(f"\n原始程序输出:\n{orig_out}")
    
    for name, mid in test_cases:
        if not (mutants_dir / f"{mid}.py").exists():
            print(f"{mid}.py 不存在，跳过")
            continue
            
        mutant_func = load_softmax_func(mutants_dir / f"{mid}.py")
        mut_out, mut_err = run_softmax_safe(mutant_func, logits, temp, axis)
        
        if mut_err:
            print(f"\n{name}: 抛出异常 → {type(mut_err).__name__}")
            print(f"  → 非等价，但原测试集未触发")
        else:
            diff = np.abs(orig_out - mut_out)
            max_diff = np.max(diff)
            print(f"\n{name}输出:\n{mut_out}")
            print(f"最大差异: {max_diff}")
            
            if max_diff < 1e-6:
                print(f"  → 输出完全相同（真等价体或测试仍不充分）")
            else:
                print(f"  → 输出不同，原测试集遗漏")


def check_abs_location(mid):
    """检查ABS变异的具体位置（修复编码问题）"""
    try:
        # 明确指定UTF-8编码
        with open(f"M{mid}.py", encoding='utf-8') as f:
            code = f.read()
        
        if "np.abs(logits)" in code:
            return "logits层（关键，有效）- 改变输入符号"
        elif "np.abs(shifted)" in code:
            return "shifted层（部分有效）- 数值稳定性后"
        elif "np.abs(scaled)" in code:
            return "scaled层（有效）- 温度缩放后"
        elif "np.abs(exp_x)" in code or "np.abs(sum_exp)" in code:
            return "exp_x层（数学等价）- exp输出恒正，abs无效"
        else:
            return "其他位置"
    except Exception as e:
        return f"读取错误: {e}"


if __name__ == "__main__":
    main4()
    # 或者：单独运行诊断（推荐先执行这个）
    # print("运行存活变异体诊断...")
    
    # # 先加载已有数据（如果你已经生成过）
    # coverage_data = generate_coverage_vectors(n_tests=500, seed=42)
    
    # # 运行诊断
    # diagnose_low_kill_mutants(mutants_dir, coverage_data)
    
    # # 快速对比ABS和LOR
    # quick_check_abs_lor(mutants_dir)

    # 检查所有ABS
    # 检查所有ABS变异体
    # print("ABS变异体位置分析：")
    # print("-" * 50)
    # for i in range(51, 61):
    #     location = check_abs_location(i)
    #     print(f"M{i}: {location}")
