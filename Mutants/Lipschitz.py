import os
import sys
import importlib.util
import pickle
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Callable

# ==================== 用户配置区（请根据实际情况修改） ====================
LHS_SUITE_PATH = "softmax/fixed_test_suite.pkl"   # LHS测试套件路径
MUTANT_DIR = "softmax/"                                         # M00.py-M63.py所在目录
VIOLATION_MAP_PATH = "softmax/vp.pkl" # 已有的违规记录文件路径（若无，代码会提示）

DELTA = 1e-6          # 输入扰动幅度。建议范围：1e-4 ~ 1e-2（根据logits量级调整）
N_DIRECTIONS = 5      # 每个输入点测试几个扰动方向（5-10次取平均，降低随机性）
RANDOM_SEED = 42
# =======================================================================

all_behavior_types = [
    "invalid_output", "nan", "inf",
    "negative_probability", "exceeds_one", "numerical_underflow",
    "probability_sum_violation", "probability_contraction",
    "uniform_distribution", "degenerate_distribution", "low_entropy_sharp",
    "high_entropy_flat", "dynamic_range_collapse",
    "monotonicity_violation", "topk_inconsistency", "broadcasting_error",
    "gradient_saturation", "gradient_explosion_risk", "extreme_temperature",
    "cross_class_violation"
]

categories = {
    'Numerical Stability': [0, 1, 2, 16, 17, 18],
    'Statistical Properties': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Semantic / Logic': [13, 14],
    'Structural / Dimension': [15, 19]
}


def load_lhs_suite(path: str) -> List[Tuple]:
    """加载LHS测试套件，假设格式为 {'test_cases': [(logits, temp, axis), ...]}"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # 兼容两种常见存储格式
    if isinstance(data, dict) and 'test_cases' in data:
        return data['test_cases']
    return data  # 如果直接存的是list


def load_mutant_func(module_name: str, mutant_dir: str = ".") -> Callable:
    """动态加载 Mxx.py 中的 stable_softmax 函数"""
    file_path = os.path.join(mutant_dir, f"{module_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到变异体文件: {file_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.stable_softmax


def safe_call(fn: Callable, logits, temp, axis):
    """安全调用变异体，若本身崩溃（如除零）则返回NaN数组"""
    try:
        return np.array(fn(logits, temp, axis), dtype=np.float64)
    except Exception:
        return np.full_like(np.array(logits, dtype=np.float64), np.nan, dtype=np.float64)


def compute_local_lipschitz(mutant_fn: Callable,
                            test_cases: List[Tuple],
                            delta: float = 1e-3,
                            n_directions: int = 5) -> Tuple[float, List[float]]:
    """
    计算变异体的最大局部Lipschitz常数（输入敏感度）。
    
    原理：对每个LHS输入x，在多个随机方向上施加微小扰动δ，
          计算 ||f(x+δ) - f(x)||_2 / ||δ||_2，取所有输入点的最大值。
          
    返回: (最大敏感度, 每个输入点的敏感度列表)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    input_lipschitz = []  # 记录每个输入点的敏感度
    
    for logits, temp, axis in test_cases:
        logits = np.array(logits, dtype=np.float64)
        
        # 在该输入点上测试多个扰动方向
        local_ls = []
        for _ in range(n_directions):
            # 生成随机方向噪声，归一化到幅度delta
            noise = rng.standard_normal(logits.shape)
            noise_flat = noise.flatten()
            norm = np.linalg.norm(noise_flat)
            if norm < 1e-12:
                continue
            noise = (noise / norm) * delta
            
            # 原始输出 vs 扰动输出
            y1 = safe_call(mutant_fn, logits, temp, axis)
            y2 = safe_call(mutant_fn, logits + noise, temp, axis)
            
            # 若输出含NaN，跳过该方向
            if np.isnan(y1).any() or np.isnan(y2).any():
                continue
            
            y_diff = np.linalg.norm(y2 - y1)
            x_diff = np.linalg.norm(noise)
            
            if x_diff > 1e-12:
                local_ls.append(y_diff / x_diff)
        
        if local_ls:
            # 该输入点多个方向的平均敏感度
            input_lipschitz.append(np.mean(local_ls))
        else:
            input_lipschitz.append(0.0)
    
    max_lipschitz = max(input_lipschitz) if input_lipschitz else 0.0
    return max_lipschitz, input_lipschitz


def compute_violation_frequency(violation_map: Dict[str, np.ndarray],
                                mutant_id: str,
                                n_cases: int) -> Tuple[float, Dict[str, float]]:
    """
    计算违规频率。
    
    返回: (总违规频率, 各层违规频率字典)
          总违规频率 = 该变异体20种违规触发次数之和 / 测试用例数
    """
    if mutant_id not in violation_map:
        return 0.0, {}
    
    vec = violation_map[mutant_id]
    if isinstance(vec, list):
        vec = np.array(vec)
    
    total_violations = np.sum(vec)
    total_freq = total_violations / n_cases
    
    # 分层频率（用于后续分析）
    layer_freq = {}
    for layer_name, indices in categories.items():
        layer_count = np.sum(vec[indices])
        layer_freq[layer_name] = layer_count / n_cases
    
    return total_freq, layer_freq


def main():
    # 1. 加载LHS测试套件
    if not os.path.exists(LHS_SUITE_PATH):
        print(f"错误：找不到LHS套件 {LHS_SUITE_PATH}")
        print("请修改代码中的 LHS_SUITE_PATH 为实际路径")
        return
    
    test_cases = load_lhs_suite(LHS_SUITE_PATH)
    n_cases = len(test_cases)
    print(f"✓ 加载LHS套件: {n_cases} 个测试用例\n")
    
    # 2. 加载违规记录
    violation_map = {}
    if os.path.exists(VIOLATION_MAP_PATH):
        with open(VIOLATION_MAP_PATH, 'rb') as f:
            violation_map = pickle.load(f)["violation_map"]
        print(f"✓ 加载违规记录: {len(violation_map)} 个变异体\n")
    else:
        print(f"⚠ 未找到违规记录文件: {VIOLATION_MAP_PATH}")
        print("  请确保你之前实验的 violation_map 已保存为pickle文件。")
        print("  格式应为: {'M01': array([0,1,0,...]), 'M02': ...} （长度20的数组）\n")
    
    # 3. 遍历 M00(参考) 和 M01-M63
    mutant_ids = ["M00"] + [f"M{i:02d}" for i in range(1, 64)]
    
    results = []
    
    print("=" * 60)
    print(f"开始计算（扰动幅度 δ={DELTA}, 方向数={N_DIRECTIONS}）")
    print("=" * 60)
    
    for mid in mutant_ids:
        try:
            fn = load_mutant_func(mid, MUTANT_DIR)
            max_lip, _ = compute_local_lipschitz(fn, test_cases, DELTA, N_DIRECTIONS)
            v_freq, v_layers = compute_violation_frequency(violation_map, mid, n_cases)
            
            results.append({
                'id': mid,
                'lipschitz': max_lip,
                'violation_freq': v_freq,
                'violation_layers': v_layers
            })
            
            # 打印关键信息
            layer_str = ", ".join([f"{k}={v:.2f}" for k, v in v_layers.items()])
            print(f"{mid}: Lipschitz={max_lip:.4f}, ViolationFreq={v_freq:.2f}  [{layer_str}]")
            
        except Exception as e:
            print(f"{mid}: 失败 - {e}")
    
    # 4. 统计相关性分析（排除M00，因为参考算子通常无违规）
    test_ids = [r for r in results if r['id'] != 'M00']
    
    if len(test_ids) >= 3:
        lipschitz_scores = [r['lipschitz'] for r in test_ids]
        violation_freqs = [r['violation_freq'] for r in test_ids]
        
        # Pearson 相关系数
        r, p_value = pearsonr(lipschitz_scores, violation_freqs)
        
        print("\n" + "=" * 60)
        print("统计相关性分析（M01-M63）")
        print("=" * 60)
        print(f"样本数: {len(test_ids)}")
        print(f"Pearson r = {r:.4f}")
        print(f"p-value   = {p_value:.6f}")
        
        # 显著性判断
        if p_value < 0.001:
            sig = "*** 高度显著 (p<0.001)"
        elif p_value < 0.01:
            sig = "** 显著 (p<0.01)"
        elif p_value < 0.05:
            sig = "* 边缘显著 (p<0.05)"
        else:
            sig = "不显著"
        print(f"显著性: {sig}")
        
        # 相关强度
        abs_r = abs(r)
        if abs_r > 0.7:
            strength = "强相关（支持变异体有效模拟上游极端输入）"
        elif abs_r > 0.4:
            strength = "中等相关"
        else:
            strength = "弱相关（建议检查delta取值或违规检测逻辑）"
        print(f"相关强度: {strength}")
        
        # 论文可用结论
        print("\n【论文可用表述】")
        if abs_r > 0.5 and p_value < 0.01:
            print(f'"变异体的最大局部Lipschitz常数与其违规触发频率呈显著正相关')
            print(f' (Pearson r={r:.2f}, p<0.01)，表明参数扰动有效放大了算子')
            print(f' 在输入敏感度区域的数值脆弱性，支持变异体作为上游极端')
            print(f' 输入边界采样器的有效性。"')
        
        # 5. 分层分析（额外洞察）
        print("\n" + "=" * 60)
        print("分层违规频率 vs Lipschitz 相关性（辅助分析）")
        print("=" * 60)
        for layer_name in categories.keys():
            layer_freqs = [r['violation_layers'].get(layer_name, 0) for r in test_ids]
            if np.std(layer_freqs) > 1e-9:  # 避免全0导致nan
                r_layer, p_layer = pearsonr(lipschitz_scores, layer_freqs)
                print(f"{layer_name:20s}: r={r_layer:+.3f}, p={p_layer:.4f}")
        
    else:
        print("\n有效样本不足，无法计算相关系数。请检查 violation_map 数据。")


if __name__ == "__main__":
    main()