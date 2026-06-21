
import numpy as np
from typing import Dict, List, Tuple, Optional

#region RQ4 baseline，kill greedy
def kill_greedy_selection(kill_matrix: np.ndarray, 
                          mutant_ids: List[str], 
                          budget_ratio: float) -> List[str]:
    """
    标准 Kill-Greedy 算法：基于集合覆盖的贪心选择
    
    每次选择能覆盖最多"尚未被杀死"测试用例的变异体
    
    Args:
        kill_matrix: 二值矩阵 (n_mutants × n_tests)，1表示该变异体在此测试下被杀死
        mutant_ids: 变异体ID列表，与kill_matrix行对应
        budget_ratio: 选择比例，如0.2表示选择20%（RR=80%）
    
    Returns:
        selected_ids: 选中的变异体ID列表
    """
    n_mutants = len(mutant_ids)
    n_tests = kill_matrix.shape[1]
    budget = max(1, int(n_mutants * budget_ratio))
    
    # 未被覆盖的测试用例索引集合（初始为全部）
    uncovered_tests = set(range(n_tests))
    selected_ids = []
    available_mutants = set(range(n_mutants))
    
    while len(selected_ids) < budget and available_mutants and uncovered_tests:
        # 计算每个可用变异体能覆盖的未覆盖测试用例数
        max_cover = -1
        best_mutant_idx = None
        
        for idx in available_mutants:
            # 该变异体能杀死的测试用例
            killed_by_this = set(np.where(kill_matrix[idx] == 1)[0])
            # 与未覆盖集合的交集
            cover_count = len(killed_by_this & uncovered_tests)
            
            if cover_count > max_cover:
                max_cover = cover_count
                best_mutant_idx = idx
        
        # 如果无法提升覆盖（max_cover=0），也继续选择（避免死循环）
        if best_mutant_idx is None or max_cover == 0:
            # 无法提升时，选择kill总数最多的作为备选
            best_mutant_idx = max(available_mutants, 
                                key=lambda x: kill_matrix[x].sum())
        
        # 选中该变异体
        selected_ids.append(mutant_ids[best_mutant_idx])
        available_mutants.remove(best_mutant_idx)
        
        # 更新未覆盖集合（移除被该变异体覆盖的测试用例）
        killed_by_selected = set(np.where(kill_matrix[best_mutant_idx] == 1)[0])
        uncovered_tests -= killed_by_selected
    
    return selected_ids

# 使用示例：
# selected = kill_greedy_selection_debug(kill_matrix_array, mutant_ids, 0.2)
def calculate_upc(selected_ids: List[str],
                  violation_map: Dict[str, List[int]], 
                  all_behavior_types: List[str]) -> float:
    """
    计算选中变异体集合的 Upstream Perturbation Coverage (UPC)
    
    Args:
        selected_ids: 选中的变异体ID列表（来自kill_greedy_selection）
        violation_map: 字典 {mutant_id: [count_v1, count_v2, ...]}
        all_behavior_types: 违规类型名称列表，长度m
    
    Returns:
        upc: 归一化香农熵，范围[0,1]
    """
    m = len(all_behavior_types)
    if m <= 1:
        return 1.0 if m == 1 else 0.0
    
    # 统计选中集的违规触发频次
    viol_counts = {vt: 0 for vt in all_behavior_types}
    
    for mid in selected_ids:
        if mid in violation_map:
            vec = violation_map[mid]
            for idx, count in enumerate(vec):
                if idx < len(all_behavior_types):
                    viol_counts[all_behavior_types[idx]] += count
    
    values = np.array(list(viol_counts.values()), dtype=float)
    total = np.sum(values)
    
    if total == 0:
        return 0.0
    
    # 计算概率分布和香农熵
    probs = values / total
    nonzero_probs = probs[probs > 0]
    
    if len(nonzero_probs) == 0:
        return 0.0
    
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    # 归一化到[0,1]
    upc = entropy / np.log(m)
    return float(np.clip(upc, 0.0, 1.0))

def calculate_ver(selected_upc: float, full_upc: float) -> float:
    """
    计算 Violation Entropy Retention (VER)
    
    Args:
        selected_upc: 选中集的UPC
        full_upc: 全集的UPC（基线）
    
    Returns:
        ver: 百分比，如95.0表示保留了95%的多样性
    """
    if full_upc == 0:
        return 0.0
    return (selected_upc / full_upc) * 100.0

#endregion
