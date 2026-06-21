import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import defaultdict, Counter
import warnings
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, cdist


#region 计算同kill异违规类型碰撞试验
def detect_fingerprint_collisions(kill_matrix, violation_map):
    """
    检测两种向量不匹配情况：
    1. 相同kill向量 → 不同多样性向量 (信息丢失)
    """
    from collections import defaultdict
    print(f"\n kill_matrix数量= {len(kill_matrix)} ")
    kill_to_group = defaultdict(list)      
    
    for name in kill_matrix.keys():
        kill_tuple = tuple(kill_matrix[name].astype(int))
        div_tuple = tuple(violation_map[name].round(6))
        
        # 保存 (变异体名, 多样性向量) 元组
        kill_to_group[kill_tuple].append((name, div_tuple))
    print(f"\n kill_to_group数量= {len(kill_to_group)} ")
    case1_collisions = []
    for kill_vec, items in kill_to_group.items():
        unique_diversities = set([item[1] for item in items])
        if len(unique_diversities) > 1:
            case1_collisions.append({
                'kill_vec': kill_vec,
                'kill_count': sum(kill_vec),
                'mutant_count': len(items),
                'mutants': [item[0] for item in items],
                'diversity_variants': list(unique_diversities),
                # 新增：保存完整映射关系用于详细打印
                'mutant_details': items
            })
    
    return case1_collisions

def print_collision_report(case1_results, total_mutants):
    """打印详细的碰撞检测报告"""
    print("=" * 70)
    print("变异体指纹碰撞检测报告")
    print("=" * 70)
    
    # 情况1
    print(f"\n【类型1】Kill向量相同但输出多样性不同 (发现 {len(case1_results)} 组)")
    print("-" * 70)
    print("说明: 这些变异体被完全相同的测试用例杀死，但展现出不同的行为模式")
    
    for i, group in enumerate(case1_results, 1):
        print(f"\n组 {i}: {group['mutant_count']} 个变异体 | "
              f"Kill数: {group['kill_count']}/{len(group['kill_vec'])} | "
              f"MS: {group['kill_count']/len(group['kill_vec']):.2f}")
        print(f"涉及变异体: {', '.join(group['mutants'])}")
        
        # Kill向量打印（转换numpy类型为纯int）
        kv = [int(x) for x in group['kill_vec']]
        print(f"共同Kill向量: {kv[:50]}..." if len(kv) > 50 else f"共同Kill向量: {kv}")
        
        # 多样性向量统计
        divs = group['diversity_variants']
        print(f"多样性向量差异数: {len(divs)} 种唯一模式")
        
        # 详细列表：带序号的每个变异体的多样性向量
        print(f"\n  详细行为多样性矩阵 (共 {group['mutant_count']} 个变异体):")
        print("  " + "-" * 60)
        
        for idx, (mutant_name, div_tuple) in enumerate(group['mutant_details'], 1):
            # 转换为标准Python整数列表，避免np.int64显示
            dv = [int(x) for x in div_tuple]
            # 打印序号、变异体名、向量值
            print(f"  [{idx:2d}] {mutant_name:10s}: {dv}")
        
        # 可选：如果需要查看唯一的多样性模式汇总，可取消下面注释
        # print(f"\n  唯一多样性模式汇总:")
        # for j, div in enumerate(divs, 1):
        #     dv = [int(x) for x in div]
        #     print(f"    模式{j}: {dv}")

    # 统计摘要
    print("\n" + "=" * 70)
    print("摘要统计")
    print("-" * 70)
    all_affected = set()
    for g in case1_results:
        all_affected.update(g['mutants'])
    
    print(f"同kill向量碰撞涉及变异体数: {sum(g['mutant_count'] for g in case1_results)}")
    print(f"总碰撞涉及变异体数(去重): {len(all_affected)}/{total_mutants}")
    print(f"总碰撞率: {len(all_affected) / total_mutants * 100:.1f}%")    

import numpy as np
from collections import defaultdict

def detect_fingerprint_collisions2(kill_matrix, violation_map):
    """
    量化 diversity 相对于 kill matrix 的分辨率优势。
    等价变异体作为 ground truth 验证 diversity 不引入假差异。
    """
    # 1. 分离等价与非等价
    eq_names = [n for n, k in kill_matrix.items() if np.all(np.asarray(k) == 0)]
    non_eq_names = [n for n in kill_matrix if n not in eq_names]
    
    print("=" * 50)
    print("【等价变异体验证】(Ground Truth: 24个完全等价)")
    
    if len(eq_names) == 0:
        print("  无等价变异体")
    else:
        eq_divs = set(tuple(np.asarray(violation_map[n]).round(6)) for n in eq_names)
        eq_kills = set(tuple(np.asarray(kill_matrix[n]).astype(int)) for n in eq_names)
        
        print(f"  数量: {len(eq_names)}")
        print(f"  kill matrix 唯一向量: {len(eq_kills)} (预期: 1)")
        print(f"  diversity 唯一向量: {len(eq_divs)} (预期: 1)")
        
        if len(eq_divs) == 1:
            print("  ✅ diversity 对等价体识别正确，无假差异")
        else:
            print(f"  ⚠️  diversity 引入 {len(eq_divs)} 种假差异")
    
    print("-" * 50)
    print("【非等价变异体分辨率对比】")
    
    if len(non_eq_names) == 0:
        print("  无非等价变异体")
        return
    
    # 提取非等价组的向量
    non_eq_kills = [tuple(np.asarray(kill_matrix[n]).astype(int)) for n in non_eq_names]
    non_eq_divs = [tuple(np.asarray(violation_map[n]).round(6)) for n in non_eq_names]
    
    unique_kills = len(set(non_eq_kills))
    unique_divs = len(set(non_eq_divs))
    n_non = len(non_eq_names)
    
    # 核心指标：分辨率增益
    gain = unique_divs / unique_kills if unique_kills > 0 else 0
    
    print(f"  非等价变异体: {n_non} 个")
    print(f"  kill matrix 唯一向量: {unique_kills} (压缩率: {unique_kills/n_non:.1%})")
    print(f"  diversity 唯一向量: {unique_divs} (分辨率: {unique_divs/n_non:.1%})")
    print(f"  📈 分辨率增益: {gain:.2f}x")
    print(f"     → diversity 的区分能力是 kill matrix 的 {gain:.1f} 倍")
    
    # 3. kill 同组内的 diversity 细分能力
    kill_groups = defaultdict(list)
    for n in non_eq_names:
        k = tuple(np.asarray(kill_matrix[n]).astype(int))
        d = tuple(np.asarray(violation_map[n]).round(6))
        kill_groups[k].append(d)
    
    multi_div_groups = 0
    total_subclasses = 0
    group_details = []
    
    for k, divs in kill_groups.items():
        u = len(set(divs))
        total_subclasses += u
        if u > 1:
            multi_div_groups += 1
            group_details.append(f"    kill组({sum(k)} kills): {len(divs)}个变异体 → {u}种diversity")
    
    print("-" * 50)
    print("【kill 同组内的 diversity 细分】")
    print(f"  kill 分组总数: {len(kill_groups)}")
    print(f"  存在 diversity 细分的组: {multi_div_groups}")
    print(f"  平均每个 kill 组含 diversity 子类: {total_subclasses/len(kill_groups):.2f}")
    
    if multi_div_groups > 0 and multi_div_groups <= 5:
        for line in group_details:
            print(line)
    elif multi_div_groups > 5:
        for line in group_details[:3]:
            print(line)
        print(f"    ... 等共 {multi_div_groups} 个组")
    
    print("=" * 50)
    print("【论文可用结论】")
    print(f"  1. 等价体识别: diversity 正确识别 {len(eq_names)} 个等价体为单一语义类")
    print(f"  2. 分辨率增益: 非等价组中 diversity 提供 {gain:.2f}x 于 kill matrix 的区分粒度")
    print(f"  3. 细粒度审计: {multi_div_groups} 个 kill 不可区分的组内，diversity 进一步分出违规子类")
    
    return {
        'equivalent_count': len(eq_names),
        'equivalent_div_unique': len(eq_divs) if eq_names else 0,
        'non_equivalent_count': n_non,
        'unique_kills': unique_kills,
        'unique_divs': unique_divs,
        'resolution_gain': float(gain),
        'kill_groups_total': len(kill_groups),
        'kill_groups_multi_div': multi_div_groups,
        'avg_subclasses_per_kill_group': total_subclasses / len(kill_groups) if kill_groups else 0
    }


# ========== 调用 ==========
# result = analyze_granularity_advantage(kill_matrix, violation_map)

# ========== 调用示例 ==========
# result = analyze_semantic_loss(kill_matrix, violation_map, metric='hamming')

# ========== 使用示例 ==========
# result = audit_kill_matrix_distortion(kill_matrix, violation_map, metric='hamming')
# print(result['interpretation'])
# print(f"同组平均距离: {result['same_kill_group']['mean_dist']:.2f}")
# print(f"不同组平均距离: {result['diff_kill_group']['mean_dist']:.2f}")

def print_collision_report2(all_same_kill_groups, total_mutants):  
    """
    all_same_kill_groups: 所有 Kill 向量相同的组（含多样性相同的组）。
      每组需包含: 'mutants': [name, ...], 'mutant_details': [(name, div_vec), ...]
    total_mutants: 该算子变异体总数
    """
    import math
    try:
        from scipy import stats
    except ImportError:
        stats = None

    total_same_kill_pairs = 0
    redundant_pairs = 0
    affected_mutants = set()
    group_count = 0

    for group in all_same_kill_groups:
        n = len(group.get('mutants', []))
        if n < 2:
            continue

        group_count += 1
        affected_mutants.update(group['mutants'])
        same_kill = n * (n - 1) // 2
        total_same_kill_pairs += same_kill

        div_vecs = []
        for detail in group.get('mutant_details', []):
            div_vecs.append(tuple(int(x) for x in detail[1]))

        diff_count = 0
        L = len(div_vecs)
        for i in range(L):
            for j in range(i + 1, L):
                if div_vecs[i] != div_vecs[j]:
                    diff_count += 1
        redundant_pairs += diff_count

    rmr = redundant_pairs / total_same_kill_pairs if total_same_kill_pairs > 0 else 0.0

    p0 = 0.2
    if total_same_kill_pairs > 0 and stats:
        se_null = math.sqrt(p0 * (1 - p0) / total_same_kill_pairs)
        z_stat = (rmr - p0) / se_null if se_null > 0 else 0.0
        p_value = 1 - stats.norm.cdf(z_stat)

        # Wilson Score Interval (小样本更稳健)
        z = 1.96
        n_pairs = total_same_kill_pairs
        x = redundant_pairs
        center = (x + z * z / 2) / (n_pairs + z * z)
        half = z * math.sqrt((x * (n_pairs - x) / n_pairs + z * z / 4) / (n_pairs + z * z))
        ci_lower = max(0.0, center - half)
        ci_upper = min(1.0, center + half)
    else:
        z_stat = p_value = ci_lower = ci_upper = float('nan')

    print("=" * 70)
    print("RMR 统计与显著性检验报告")
    print("=" * 70)
    print(f"Kill向量相同组数:        {group_count}")
    print(f"Kill向量相同配对总数(N): {total_same_kill_pairs}")
    print(f"冗余误判配对数(X):       {redundant_pairs}")
    print(f"RMR = X/N:               {rmr:.4f} ({rmr*100:.2f}%)")
    print("-" * 70)
    print("单样本比例检验 (H0: RMR <= 0.2, H1: RMR > 0.2)")
    if not math.isnan(z_stat):
        print(f"  Z 统计量:              {z_stat:.4f}")
        print(f"  单侧 P 值:             {p_value:.4e}")
        print(f"  95% Wilson CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  显著性标记:            {sig}")
        print(f"  结论:                  {'拒绝 H0, RMR 显著高于 0.2' if p_value < 0.05 else '不拒绝 H0'}")
    else:
        print("  统计量:                N/A")
    print("-" * 70)
    print(f"涉及变异体数(去重):      {len(affected_mutants)}/{total_mutants}")
    print(f"碰撞覆盖率:              {len(affected_mutants)/total_mutants*100:.1f}%")
    print("=" * 70)

#endregion


#region 分类相关性检验
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy import stats
import random


def analyze_single_operator(
    op_name,
    kill_matrix,
    violation_map,
    metric='hamming',
    use_independent_sample: bool = True,   # 开启无重叠独立配对（防伪重复）
    n_sample_round: int = 20,              # 抽样总轮数，可按需调大(30/50/100)
    random_seed: int = 42                  # 固定种子，结果可复现
):
    """
    单算子分析 + 多轮独立无重叠配对累积统计
    修复伪重复 + 累积多轮样本提升McNemar检验效力
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 1. 分离等价 / 非等价变体（原有逻辑不变）
    eq_names = [n for n, k in kill_matrix.items() if np.all(np.asarray(k) == 0)]
    non_eq_names = [n for n in kill_matrix if n not in eq_names]
    n_eq, n_non = len(eq_names), len(non_eq_names)

    # 2. 等价体指纹统计（不变）
    eq_div_unique = 0
    if n_eq >= 2:
        eq_divs = set(tuple(np.asarray(violation_map[n]).round(6)) for n in eq_names)
        eq_div_unique = len(eq_divs)

    # 3. 非等价组分辨率统计（不变）
    non_eq_kills = [tuple(np.asarray(kill_matrix[n]).astype(int)) for n in non_eq_names]
    non_eq_divs = [tuple(np.asarray(violation_map[n]).round(6)) for n in non_eq_names]
    unique_kills = len(set(non_eq_kills))
    unique_divs = len(set(non_eq_divs))
    gain = unique_divs / unique_kills if unique_kills > 0 else 0.0

    # ===================== 核心：多轮抽样 累积所有独立配对 =====================
    if not use_independent_sample:
        # 模式1：原始全量两两组合（对比用，存在伪重复）
        both_diff = kill_only = div_only = both_same = 0
        total_pairs = 0
        for m1, m2 in combinations(non_eq_names, 2):
            total_pairs += 1
            k_same = np.array_equal(np.asarray(kill_matrix[m1]), np.asarray(kill_matrix[m2]))
            d1 = np.asarray(violation_map[m1]).round(6)
            d2 = np.asarray(violation_map[m2]).round(6)
            d_same = np.array_equal(d1, d2)

            if not k_same and not d_same:
                both_diff += 1
            elif not k_same and d_same:
                kill_only += 1
            elif k_same and not d_same:
                div_only += 1
            else:
                both_same += 1
    else:
        # 模式2：多轮独立无重叠配对 -> 累积总频数（推荐）
        both_diff = kill_only = div_only = both_same = 0
        total_pairs = 0

        for _ in range(n_sample_round):
            shuffled = non_eq_names.copy()
            random.shuffle(shuffled)
            # 生成本轮无重叠配对
            paired = []
            for i in range(0, len(shuffled) - 1, 2):
                paired.append((shuffled[i], shuffled[i+1]))

            # 逐对统计并累加到全局
            for m1, m2 in paired:
                total_pairs += 1
                k_same = np.array_equal(np.asarray(kill_matrix[m1]), np.asarray(kill_matrix[m2]))
                d1 = np.asarray(violation_map[m1]).round(6)
                d2 = np.asarray(violation_map[m2]).round(6)
                d_same = np.array_equal(d1, d2)

                if not k_same and not d_same:
                    both_diff += 1
                elif not k_same and d_same:
                    kill_only += 1
                elif k_same and not d_same:
                    div_only += 1
                else:
                    both_same += 1
    # =========================================================================

    # 4. McNemar 精确二项检验（基于累积后的总分歧样本）
    discordant = kill_only + div_only
    p_mcnemar = 1.0
    if discordant > 0:
        p_mcnemar = stats.binomtest(div_only, discordant, p=0.5, alternative='greater').pvalue

    # 5. Kill分组细分统计（不变）
    kill_groups = defaultdict(list)
    for n in non_eq_names:
        k = tuple(np.asarray(kill_matrix[n]).astype(int))
        d = tuple(np.asarray(violation_map[n]).round(6))
        kill_groups[k].append(d)

    multi_div_groups = sum(1 for divs in kill_groups.values() if len(set(divs)) > 1)

    return {
        'operator': op_name,
        'n_total': len(kill_matrix),
        'n_eq': n_eq,
        'eq_fingerprint_ok': (eq_div_unique == 1) if n_eq >= 2 else None,
        'n_non': n_non,
        'unique_kills': unique_kills,
        'unique_divs': unique_divs,
        'resolution_gain': float(gain),
        'both_diff': both_diff,
        'kill_only': kill_only,
        'div_only': div_only,
        'both_same': both_same,
        'discordant': discordant,
        'p_mcnemar': float(p_mcnemar),
        'multi_div_groups': multi_div_groups,
        'kill_groups_total': len(kill_groups),
        'sample_rounds': n_sample_round if use_independent_sample else None,
        'total_pairs': total_pairs  # 新增：总独立配对数，方便核对
    }




def print_operator_summary(r):
    """打印单个算子结果"""
    print(f"\n{'='*50}")
    print(f"算子: {r['operator']}")
    print(f"  总变异体: {r['n_total']} | 等价体: {r['n_eq']} | 非等价: {r['n_non']}")
    if r['eq_fingerprint_ok'] is not None:
        flag = "✅" if r['eq_fingerprint_ok'] else "❌"
        print(f"  等价体指纹: {flag} ({r['n_eq']}个等价体 → {r['eq_fingerprint_ok']})")
    print(f"  分辨率: kill={r['unique_kills']} vs div={r['unique_divs']} (增益={r['resolution_gain']:.2f}x)")
    print(f"  McNemar: div_only={r['div_only']}, kill_only={r['kill_only']}, "
          f"both_diff={r['both_diff']}, both_same={r['both_same']}")
    print(f"  配对检验 p={r['p_mcnemar']:.4f}")
    print(f"  kill同组细分: {r['multi_div_groups']}/{r['kill_groups_total']} 组")


def meta_analysis(results):
    """
    跨算子元分析。
    results: list[dict]，由 analyze_single_operator 输出组成。
    """
    print(f"\n{'='*60}")
    print("【跨算子元分析】")
    
    k = len(results)
    if k == 0:
        print("无算子数据")
        return
    
    # 1. 描述性汇总
    gains = [r['resolution_gain'] for r in results if r['resolution_gain'] > 0]
    pvals = [r['p_mcnemar'] for r in results if r['discordant'] > 0]
    
    print(f"  算子数量: {k}")
    print(f"  分辨率增益范围: {min(gains):.2f}x ~ {max(gains):.2f}x (中位数={np.median(gains):.2f})")
    print(f"  等价体指纹正确: {sum(1 for r in results if r['eq_fingerprint_ok'] is True)}/{k}")
    
    # 2. 符号检验：多少算子分辨率增益 > 1.0
    positive = sum(1 for g in gains if g > 1.0)
    print(f"  增益>1.0的算子: {positive}/{len(gains)}")
    if len(gains) > 0:
        p_sign = stats.binomtest(positive, len(gains), p=0.5, alternative='greater').pvalue
        print(f"  符号检验 p={p_sign:.4f} (H1: 多数算子增益>1)")
    
    # 3. Fisher's Method 合并 p 值（仅取 discordant>0 的算子）
    valid_p = [p for p in pvals if 0 < p < 1]
    if len(valid_p) > 1:
        chi2 = -2 * sum(np.log(p) for p in valid_p)
        df = 2 * len(valid_p)
        p_combined = 1 - stats.chi2.cdf(chi2, df)
        print(f"\n  Fisher合并检验:")
        print(f"    纳入算子: {len(valid_p)}")
        print(f"    χ²={chi2:.2f}, df={df}, 合并p={p_combined:.4f}")
        if p_combined < 0.01:
            print("    → 跨算子整体显著：diversity 提供显著更多区分")
        elif p_combined < 0.05:
            print("    → 跨算子整体显著 (p<<0.05)")
        else:
            print("    → 跨算子整体不显著")
    
    # 4. 各算子 McNemar 显著性汇总
    sig = sum(1 for p in valid_p if p < 0.05)
    print(f"\n  单个算子显著(p<<0.05): {sig}/{len(valid_p)}")
    
    print(f"{'='*60}")
    
    return {
        'n_operators': k,
        'median_gain': float(np.median(gains)),
        'gain_positive_ratio': positive / len(gains) if gains else 0,
        'p_sign_test': float(p_sign) if gains else 1.0,
        'p_fisher_combined': float(p_combined) if len(valid_p) > 1 else 1.0,
        'significant_operators': sig
    }


# ========== 使用示例 ==========
# results = []
# for op_name, (km, vm) in operators_data.items():  # 你的多算子数据字典
#     r = analyze_single_operator(op_name, km, vm)
#     print_operator_summary(r)
#     results.append(r)
# 
# meta = meta_analysis(results)

#endregion

#region Kill 同组条件置换检验
def conditional_diversity_test(kill_matrix, violation_map, n_perm=10000, seed=42):
    """
    Kill同组条件置换检验：
    检验在kill相同的配对中，diversity的区分是否显著多于随机基线。
    不受测试用例数量影响。
    """
    rng = np.random.default_rng(seed)
    names = list(kill_matrix.keys())
    n = len(names)
    
    # 提取为numpy数组加速
    kills = np.array([np.asarray(kill_matrix[name]).astype(int) for name in names])
    divs = [np.asarray(violation_map[name]).round(6) for name in names]
    
    # 预计算kill相等矩阵（布尔，上三角）
    kill_equal = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            kill_equal[i, j] = np.array_equal(kills[i], kills[j])
    
    # 观察值：kill同组且diversity不同的配对数
    real_stat = 0
    for i in range(n):
        for j in range(i+1, n):
            if kill_equal[i, j] and not np.array_equal(divs[i], divs[j]):
                real_stat += 1
    
    # 快速路径：如果观察值为0，无需置换
    if real_stat == 0:
        print("观察值为0，kill同组内无diversity差异")
        return {'observed': 0, 'p_value': 1.0, 'mean_fake': 0.0}
    
    # 置换检验：只重排diversity，kill分组不变
    fake_stats = np.zeros(n_perm, dtype=int)
    div_arrays = [np.array(d) for d in divs]  # 转为统一类型便于比较
    
    for p in range(n_perm):
        idx = rng.permutation(n)
        fake_divs = [div_arrays[i] for i in idx]
        
        cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                if kill_equal[i, j] and not np.array_equal(fake_divs[i], fake_divs[j]):
                    cnt += 1
        fake_stats[p] = cnt
    
    p_value = np.mean(fake_stats >= real_stat)
    
    print("=" * 50)
    print("【Kill同组条件置换检验】")
    print(f"  变异体数: {n}")
    print(f"  kill同组总配对: {int(kill_equal.sum())}")
    print(f"  观察值(kill同组且div不同): {real_stat}")
    print(f"  随机基线均值: {fake_stats.mean():.1f} ± {fake_stats.std():.1f}")
    print(f"  置换检验 p-value: {p_value:.4f}")
    print(f"  → {'高度显著' if p_value < 0.01 else '显著' if p_value < 0.05 else '不显著'} "
          f"(H0: kill同组内diversity随机分配)")
    print("=" * 50)
    
    return {
        'observed': int(real_stat),
        'n_kill_equal_pairs': int(kill_equal.sum()),
        'p_value': float(p_value),
        'mean_fake': float(fake_stats.mean()),
        'std_fake': float(fake_stats.std()),
        'fake_distribution': fake_stats.tolist()
    }


# ========== 辅助：分辨率增益的稳定性检验 ==========
def resolution_gain_test(kill_matrix, violation_map, n_perm=10000, seed=42):
    """
    检验观察到的分辨率增益(unique_divs/unique_kills)是否显著大于随机基线。
    零假设：diversity向量随机分配时，增益分布与观察值无差异。
    """
    rng = np.random.default_rng(seed)
    names = list(kill_matrix.keys())
    
    # 观察值
    real_kills = [tuple(np.asarray(kill_matrix[n]).astype(int)) for n in names]
    real_divs = [tuple(np.asarray(violation_map[n]).round(6)) for n in names]
    real_gain = len(set(real_divs)) / len(set(real_kills)) if real_kills else 0
    
    # 置换：随机重排diversity，计算增益分布
    fake_gains = []
    for _ in range(n_perm):
        shuffled = real_divs.copy()
        rng.shuffle(shuffled)
        g = len(set(shuffled)) / len(set(real_kills))
        fake_gains.append(g)
    
    fake_gains = np.array(fake_gains)
    p_value = np.mean(fake_gains >= real_gain)
    
    print("=" * 50)
    print("【分辨率增益置换检验】")
    print(f"  观察增益: {real_gain:.3f}x")
    print(f"  随机基线: {fake_gains.mean():.3f}x ± {fake_gains.std():.3f}")
    print(f"  p-value:  {p_value:.4f}")
    print(f"  → {'显著' if p_value < 0.05 else '不显著'} (H0: 增益由随机产生)")
    print("=" * 50)
    
    return {
        'observed_gain': float(real_gain),
        'p_value': float(p_value),
        'mean_fake_gain': float(fake_gains.mean())
    }
#endregion


def run_optimized_analysis(kill_matrix, violation_map, ms_per_mutant, 
                          behavior_types, operator_map, budget_ratio=0.2, 
                          force_subdivide=False,title=''):
    """
    基于行为指纹的算子优先聚类分析
    
    Parameters:
    -----------
    kill_matrix : dict
        变异体的kill矩阵 {mutant_id: kill_vector}
    violation_map : dict  
        行为指纹向量 {mutant_id: behavior_vector}
    ms_per_mutant : dict
        变异体得分 {mutant_id: ms_score}
    behavior_types : list
        所有可能的行为类型
    operator_map : dict
        变异体到算子类型的映射
    budget_ratio : float
        选择预算比例
    force_subdivide : bool
        是否强制细分大簇
        
    Returns:
    --------
    selected : list
        选中的变异体ID列表
    results : dict
        聚类质量指标
    details : dict
        每个簇的详细信息
    """
    
    mutant_ids = list(violation_map.keys())
    n_mutants = len(mutant_ids)
    budget = max(1, int(n_mutants * budget_ratio))
    
    # 构建特征矩阵 (行为指纹向量)
    X = np.array([violation_map[mid] for mid in mutant_ids])
    print(X)
    # exit()
    # 确定最佳聚类数 (使用肘部法则或固定范围)
    max_k = min(15, n_mutants // 2)
    if max_k < 2:
        max_k = 2
        
    best_k = 2
    best_score = -1
    
    print(f"[CLUSTER] 测试聚类范围 k=2 到 {max_k}")
    
    for k in range(2, max_k + 1):
        if k >= n_mutants:
            break
        try:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X)
            if len(set(labels_temp)) > 1:
                score = silhouette_score(X, labels_temp)
                print(f"  k={k}: Silhouette={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception as e:
            continue
    
    print(f"[CLUSTER] 选择最佳 k={best_k} (Silhouette={best_score:.3f})")
    best_k=5
    # 执行最终聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # === RQ2: 基线对比分析 ===
    metrics_v, metrics_k, labels_v, labels_k = compare_with_kill_matrix(
        violation_map, kill_matrix, mutant_ids, best_k
    )
    # === RQ2: 降维显示 ===
    plot_tsne_comparison(violation_map, kill_matrix, mutant_ids,cluster_labels,title=title)

    print("\n[COMPARISON: Behavioral vs Kill Matrix]")
    print(f"Behavioral - Silhouette: {metrics_v['silhouette']:.3f}, DB: {metrics_v['db_index']:.3f}")
    print(f"KillMatrix - Silhouette: {metrics_k['silhouette']:.3f}, DB: {metrics_k['db_index']:.3f}")

    # === Behavioral Fingerprint 距离 ===
    dist_metrics_behavior = compute_cluster_distances(X, cluster_labels)

    # === Kill Matrix 距离 ===
    X_kill = np.array([kill_matrix[mid] for mid in mutant_ids])

    kmeans_k = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(X_kill)

    dist_metrics_kill = compute_cluster_distances(X_kill, labels_k)

    # === 输出 ===
    print("\n[Distance Analysis: Behavioral]")
    for k, v in dist_metrics_behavior.items():
        print(f"{k}: {v:.4f}")

    print("\n[Distance Analysis: Kill Matrix]")
    for k, v in dist_metrics_kill.items():
        print(f"{k}: {v:.4f}")
    
    cluster_centers = kmeans.cluster_centers_
    
    # 计算聚类质量指标
    try:
        sil_score = silhouette_score(X, cluster_labels)
        db_score = davies_bouldin_score(X, cluster_labels)
    except Exception as e:
        sil_score = 0.0
        db_score = float('inf')
        print(f"[WARNING] 指标计算失败: {e}")
    
    # 按簇组织变异体
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(mutant_ids[idx])

    # 算子优先级定义 (根据常见故障检测能力排序)
    operator_priority = {
        'ROR': 1, 'COR': 2, 'AOR': 3, 'LOR': 4, 
        'UOI': 5, 'SDL': 6, 'ABS': 7, 'ORIG': 8
    }
    
    # 计算每个簇的预算分配 (基于簇内故障类型多样性)
    cluster_diversity = {}
    total_diversity = 0
    
    for cid, members in clusters.items():
        # 统计该簇覆盖的所有行为类型
        covered_behaviors = set()
        covered_operators = set()
        for mid in members:
            vec = violation_map[mid]
            # 找出非零的行为类型索引
            non_zero_idx = np.where(vec > 0)[0]
            for idx in non_zero_idx:
                covered_behaviors.add(behavior_types[idx])
            covered_operators.add(operator_map.get(mid, 'UNKNOWN'))
        
        diversity = len(covered_behaviors)
        cluster_diversity[cid] = {
            'diversity': diversity,
            'behaviors': covered_behaviors,
            'operators': covered_operators,
            'members': members
        }
        total_diversity += diversity
    
    # 按多样性比例分配预算
    selected = []
    cluster_details = {}
    all_selected_behaviors = set()
    all_selected_operators = set()
    
    print(f"\n[CLUSTER DETAILS] 共 {best_k} 个簇, 总预算 {budget}")
    print("=" * 80)
    
    for cid in sorted(clusters.keys()):
        info = cluster_diversity[cid]
        members = info['members']
        n_members = len(members)
        
        # 计算该簇预算 (至少选1个，如果 diversity > 0)
        if total_diversity > 0:
            cluster_budget = max(1, int(budget * info['diversity'] / total_diversity))
        else:
            cluster_budget = max(1, budget // best_k)
        
        # 如果簇很小，调整预算
        cluster_budget = min(cluster_budget, n_members)
        
        print(f"\n簇 {cid}: {n_members} 个变异体 | 预算 {cluster_budget} | 多样性 {info['diversity']}")
        print(f"  覆盖行为类型: {sorted(info['behaviors'])}")
        print(f"  涉及算子: {sorted(info['operators'])}")
        
        # 在簇内按算子优先 + 距离中心点距离选择
        candidates = []
        for mid in members:
            op = operator_map.get(mid, 'UNKNOWN')
            priority = operator_priority.get(op, 99)
            vec = violation_map[mid]
            # 计算到簇中心的距离 (越小越代表该簇)
            dist = np.linalg.norm(vec - cluster_centers[cid])
            # 计算该变异体的MS (mutation score)
            ms = ms_per_mutant.get(mid, 0)
            candidates.append((mid, priority, dist, ms, vec))
        
        # 排序: 先按算子优先级, 再按距离中心点距离(近者优先)
        candidates.sort(key=lambda x: (x[1], x[2]))
        
        # 选择
        cluster_selected = []
        cluster_behaviors = set()
        cluster_operators = set()
        
        for i in range(min(cluster_budget, len(candidates))):
            mid, prio, dist, ms, vec = candidates[i]
            cluster_selected.append(mid)
            selected.append(mid)
            
            # 更新覆盖统计
            non_zero_idx = np.where(vec > 0)[0]
            for idx in non_zero_idx:
                btype = behavior_types[idx]
                cluster_behaviors.add(btype)
                all_selected_behaviors.add(btype)
            
            op = operator_map.get(mid, 'UNKNOWN')
            cluster_operators.add(op)
            all_selected_operators.add(op)
            
            print(f"    -> 选中 {mid} (算子:{op}, 优先级:{prio}, MS:{ms:.2f}, 距中心:{dist:.2f})")
        
        # 保存簇详情
        cluster_details[cid] = {
            'size': n_members,
            'budget': cluster_budget,
            'selected': cluster_selected,
            'behaviors_covered': sorted(cluster_behaviors),
            'operators_involved': sorted(cluster_operators),
            'diversity': info['diversity'],
            'center': cluster_centers[cid].tolist()
        }
    
    print("=" * 80)
    
    # 计算 FTRR (Fault Type Retention Rate)
    # 这里将故障类型定义为: (行为类型, 算子类型) 的组合
    original_fault_types = set()
    for mid in mutant_ids:
        vec = violation_map[mid]
        op = operator_map.get(mid, 'UNKNOWN')
        non_zero_idx = np.where(vec > 0)[0]
        for idx in non_zero_idx:
            fault_type = (behavior_types[idx], op)
            original_fault_types.add(fault_type)
    
    selected_fault_types = set()
    for mid in selected:
        vec = violation_map[mid]
        op = operator_map.get(mid, 'UNKNOWN')
        non_zero_idx = np.where(vec > 0)[0]
        for idx in non_zero_idx:
            fault_type = (behavior_types[idx], op)
            selected_fault_types.add(fault_type)
    
    ftrr = len(selected_fault_types) / len(original_fault_types) * 100 if original_fault_types else 0
    
    # 计算 MS 保持率 (Mutation Score Retention)
    original_ms = np.mean([ms_per_mutant[mid] for mid in mutant_ids])
    selected_ms = np.mean([ms_per_mutant[mid] for mid in selected])
    ms_retention = (selected_ms / original_ms * 100) if original_ms > 0 else 0
    
    print(f"\n[RETENTION METRICS]")
    print(f"  FTRR (故障类型保持率): {ftrr:.1f}% ({len(selected_fault_types)}/{len(original_fault_types)})")
    print(f"  MS 保持率: {ms_retention:.1f}% ({selected_ms:.3f}/{original_ms:.3f})")
    print(f"  选中 {len(selected)}/{n_mutants} 变异体 ({len(selected)/n_mutants*100:.1f}%)")
    
    # 汇总结果
    results = {
        'n_clusters': best_k,
        'silhouette_score': float(sil_score),
        'davies_bouldin_score': float(db_score),
        'ftrr_percent': float(ftrr),
        'ms_retention_percent': float(ms_retention),
        'original_ms': float(original_ms),
        'selected_ms': float(selected_ms),
        'budget_used': len(selected),
        'total_mutants': n_mutants
    }
    
    details = {
        'clusters': cluster_details,
        'selected_mutants': selected,
        'clustering_labels': {mid: int(cluster_labels[i]) for i, mid in enumerate(mutant_ids)},
        'fault_type_breakdown': {
            'original': len(original_fault_types),
            'retained': len(selected_fault_types),
            'lost': len(original_fault_types - selected_fault_types)
        }
    }
    

    print("\n" + "=" * 80)
    print("[约简后集合覆盖分析报告]")
    
    # 表1: Violation Types 覆盖统计 (横向表格)
    print("\n>>> 表1: Violation Types 覆盖统计 (横向)")
    viol_counts = {vt: 0 for vt in behavior_types}
    for mid in selected:
        vec = violation_map[mid]
        for idx, val in enumerate(vec):
            if val > 0:
                viol_counts[behavior_types[idx]] += 1
    
    # 横向输出：标题行 + 数量行 (制表符分隔，便于复制到Excel)
    header_line = "Violation Type:\t" + "\t".join(behavior_types)
    count_line = "覆盖次数:\t" + "\t".join([str(viol_counts[vt]) for vt in behavior_types])
    print(header_line)
    print(count_line)
    
    # 未被覆盖的类型
    uncovered_viols = [vt for vt in behavior_types if viol_counts[vt] == 0]
    if uncovered_viols:
        print(f"\n⚠️  未覆盖类型 ({len(uncovered_viols)} 个): {', '.join(uncovered_viols)}")
    else:
        print("\n✓ 所有行为类型均被覆盖")
    
    # 表2: Operator Types 覆盖统计 (横向表格)
    print("\n>>> 表2: Operator Types 覆盖统计 (横向)")
    op_counter = {}
    for mid in selected:
        op = operator_map.get(mid, 'UNKNOWN')
        op_counter[op] = op_counter.get(op, 0) + 1
    
    # 获取原始存在的所有算子类型
    all_operators = sorted(list(set(operator_map.values())))
    op_header = "Operator Type:\t" + "\t".join(all_operators)
    op_count_line = "覆盖数量:\t" + "\t".join([str(op_counter.get(op, 0)) for op in all_operators])
    print(op_header)
    print(op_count_line)
    
    # 简要统计
    covered_ops = sum(1 for op in all_operators if op_counter.get(op, 0) > 0)
    print(f"\n算子覆盖总结: {covered_ops}/{len(all_operators)} 种算子类型被覆盖")
   

    return selected, results, details

#region 分布统计：statistic
def run_violation_statistc(all_behavior_types,violation_map):
    viol_covery={vt:0 for vt in all_behavior_types}
    
    for mid,vec in violation_map.items():        
        for idx,count in enumerate(vec):
            viol_covery[all_behavior_types[idx]]+=count
    arr={k:int(v) for k,v in viol_covery.items()}
    
    values=np.array(list(arr.values()))
    print(values)
    std=np.std(values)
    var=np.var(values)
    cv=std/np.mean(values)
    print(cv) #离散度，小于1

# UPC 计算
def run_upc_calculation(all_behavior_types, violation_map):
    # print(violation_map)
    """
    计算Upstream Perturbation Coverage (UPC)
    UPC = - (1/log m) * sum(p(vi) * log(p(vi)))
    其中p(vi)为第i类违规的触发概率归一化值
    
    参数:
        all_behavior_types: 违规类型列表 (list)
        violation_map: 字典 {mutant_id: [count_v1, count_v2, ...]}
    
    返回:
        upc: 归一化香农熵值，范围[0, 1]
    """
    m = len(all_behavior_types)
    
    # 统计每类违规的总触发次数
    viol_covery = {vt: 0 for vt in all_behavior_types}
    
    for mid, vec in violation_map.items():
        for idx, count in enumerate(vec):
            viol_covery[all_behavior_types[idx]] += count
    
    values = np.array(list(viol_covery.values()), dtype=float)
    total = np.sum(values)
    
    # 避免除0错误：如果没有违规触发，UPC定义为0（无覆盖）
    if total == 0:
        return 0.0
    
    # 计算概率分布 p(vi)
    probs = values / total
    
    # 计算香农熵: -sum(p * log(p))，忽略p=0的项（定义为0）
    # 使用自然对数，与归一化因子log(m)对应
    nonzero_probs = probs[probs > 0]
    if len(nonzero_probs) == 0:
        return 0.0
        
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    # 归一化: 除以log(m)，使UPC范围在[0, 1]
    # 当m=1时，log(1)=0，此时若entropy=0则UPC=1（唯一类型均匀分布）
    if m <= 1:
        return 1.0 if entropy == 0 else 0.0
    
    upc = entropy / np.log(m)
    
    # 数值稳定性处理：处理可能的微小负值或超1值
    upc = np.clip(upc, 0.0, 1.0)
    
    print(f'upc={upc}')
    return float(upc)
#endregion

#region 碰撞实验
def collid_graph_t(case1_results, total_mutants=63,  
                 all_behavior_types=None,
                 title="km_bv_collision graph",categories=None):
    """
    紧凑三框布局: A图(左,跨两行) | B图(右上)
                                 | C图(右下,带20个细类别名称)
    """
    output_filename=title #这个是准备输出后面的png文件用的。
    title=title+' Collision Analysis: Identical Kill Vectors with Divergent Behavioral Semantics'

    len_of_all_behavior_types=len(all_behavior_types)   
        
    cat_colors = {
        'Numerical Stability': '#C0392B',      # 深红
        'Statistical Properties': '#2980B9',   # 深蓝  
        'Semantic / Logic': '#8E44AD',         # 深紫
        'Structural / Dimension': '#27AE60'    # 深绿
    }

    # 数据解析 (保持原有逻辑)
    all_mutants = []
    all_behaviors = []
    group_ids = []
    group_stats = []
    
    for group_idx, group in enumerate(case1_results):
        mutant_details = group['mutant_details']
        diversity_vecs = [np.array(div) for _, div in mutant_details]
        
        diff_count = sum(
            1 for dim in range(len(diversity_vecs[0]))
            if len(set(vec[dim] for vec in diversity_vecs)) > 1
        ) if len(diversity_vecs) > 1 else 0
        
        group_stats.append({
            'mutant_count': group['mutant_count'],
            'kill_count': group['kill_count'],
            'ms': group['kill_count'] / len(group['kill_vec']),
            'diff_dimensions': diff_count,
            'kill_vec': group['kill_vec'][:30]
        })
        
        for name, div_tuple in mutant_details:
            all_mutants.append(name)
            all_behaviors.append(list(div_tuple))
            group_ids.append(group_idx)


    behavior_matrix = np.array(all_behaviors)
    n_groups = len(case1_results)
    group_colors = ['#D62728', '#1F77B4', '#2CA02C'][:n_groups]

    # 新增，原来是只能3种颜色，现在根据分组个数自动增加颜色
    n_colors_needed = n_groups
    if len(group_colors) < n_colors_needed:
        # 使用 colormap 生成额外颜色
        cmap = cm.get_cmap('tab10')  # 或其他 colormap
        extra_colors = [cmap(i) for i in np.linspace(0, 1, n_colors_needed)]
        group_colors = extra_colors  # 替换为完整颜色列表


    print(f"\n group_ids={group_ids},\n group_colors={group_colors},n_groups={n_groups} ")
    # 布局调整: 增大左列宽度以容纳旋转的文字标签
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.8, 1],    # 左列更宽，右列适中
        height_ratios=[1, 1.2],   # 右下略高以容纳20个标签
        wspace=0.15,
        hspace=0.35,
        left=0.08, right=0.9,
        top=0.85, bottom=0.15
    )

    # ========== A图: 行为指纹热力图 (左,跨两行) ==========
    ax_a = fig.add_subplot(gs[:, 0])
    
    im_a = ax_a.imshow(
        behavior_matrix, 
        aspect='auto', 
        cmap='YlOrRd',
        interpolation='nearest'
    )
    
    # 关键修改: 横轴显示完整类别名称，旋转90度垂直显示
    ax_a.set_xticks(range(len_of_all_behavior_types))
    
    # 生成缩写标签以节省空间 (取前8个字符+...)
    short_labels = []
    for name in all_behavior_types:
        if len(name) <= 10:
            short_labels.append(name)
        else:
            # 缩写: 前3字母+...+后3字母
            short_labels.append(name[:4] + ".." + name[-3:])
    
    ax_a.set_xticklabels(short_labels, rotation=90, ha='center', fontsize=5)
    ax_a.set_title('(a) Behavioral Fingerprints by Violation Type\n(Same Kill Vector, Different Semantics)', 
                   fontsize=9, fontweight='bold', pad=8)
    ax_a.set_xlabel('Violation Types (20 Categories)', fontsize=8)
    ax_a.set_ylabel('Grouped Mutants', fontsize=8)
    
    # Y轴标签
    ax_a.set_yticks(range(len(all_mutants)))
    ax_a.set_yticklabels(all_mutants, fontsize=7)
    for label in ax_a.get_yticklabels():
        label.set_horizontalalignment('right')

    # 左侧色带标识分组
    for i, gid in enumerate(group_ids):
        ax_a.add_patch(Rectangle((-0.6, i-0.5), 0.2, 1, 
                               facecolor=group_colors[gid], alpha=0.8))
    
    # 分组线
    cum = 0
    for g in group_stats[:-1]:
        cum += g['mutant_count']
        ax_a.axhline(y=cum-0.5, color='black', linewidth=1.5, linestyle='--', alpha=0.6)
    
    # 颜色条放在A图下方横向
    cbar = plt.colorbar(im_a, ax=ax_a, shrink=0.6, aspect=30, pad=0.1, orientation='horizontal')
    cbar.set_label('Violation Count', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ========== B图: 差异矩阵 (右上) ==========
    ax_b = fig.add_subplot(gs[0, 1])
    
    diff_mat = np.zeros_like(behavior_matrix)
    idx = 0
    for g in case1_results:
        n = g['mutant_count']
        if n > 1:
            gmat = behavior_matrix[idx:idx+n]
            gmean = gmat.mean(axis=0)
            for i in range(n):
                diff_mat[idx+i] = (np.abs(gmat[i] - gmean) > 0.01).astype(float)
        idx += n
    
    im_b = ax_b.imshow(diff_mat, aspect='auto', cmap='Blues', interpolation='nearest')
    ax_b.set_title('(b) Intra-Group\nVariations', fontsize=8, fontweight='bold', pad=4)
    ax_b.set_xlabel('Dim (0-19)', fontsize=7)
    ax_b.set_ylabel('')
    ax_b.set_xticks(range(0, len_of_all_behavior_types, 2))
    ax_b.set_xticklabels([str(i) for i in range(0, len_of_all_behavior_types, 2)], fontsize=6)
    ax_b.set_yticks([])
    
    # 分组线
    cum = 0
    for g in group_stats[:-1]:
        cum += g['mutant_count']
        ax_b.axhline(y=cum-0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.7)

    # ========== C图: 20个细类别名称+分类大括号 (右下) ==========
    ax_c = fig.add_subplot(gs[1, 1])
    
    # 准备数据：按类别分组显示所有20个名称
    y_pos = 0
    y_positions = []  # 记录每个类别的y中心位置
    all_labels = []   # 记录所有标签
    
    current_y = 0
    category_y_centers = {}
    
    for cat_name, indices in categories.items():
        if not indices:
            continue
            
        # 记录此类别的y轴中心（用于绘制大括号）
        cat_start_y = current_y
        color = cat_colors[cat_name]
        
        for idx in sorted(indices):
            # 显示格式: [idx] name
            label_text = f"[{idx:2d}] {all_behavior_types[idx]}"
            # 截断过长的名称
            if len(label_text) > 28:
                label_text = label_text[:25] + "..."
            
            # 绘制色块背景（表示分类）
            rect = plt.Rectangle((-0.5, current_y-0.4), 4, 0.8, 
                               facecolor=color, alpha=0.15, edgecolor=color, 
                               linewidth=1, zorder=1)
            ax_c.add_patch(rect)
            
            # 绘制文字
            ax_c.text(0, current_y, label_text, fontsize=5.5, 
                     va='center', ha='left', color='black', fontweight='normal',
                     family='monospace', zorder=2)
            
            # 在左侧绘制小色条表示类别
            ax_c.plot([-0.4, -0.1], [current_y, current_y], color=color, linewidth=3, solid_capstyle='butt')
            
            current_y += 1
        
        cat_end_y = current_y - 1
        category_y_centers[cat_name] = (cat_start_y + cat_end_y) / 2
    
    # 设置C图坐标轴
    ax_c.set_xlim(-0.5, 3.5)
    ax_c.set_ylim(-0.5, current_y - 0.5)
    ax_c.invert_yaxis()  # 0在上，与A图对应
    ax_c.axis('off')     # 关闭坐标轴
    
    # 绘制大括号和类别名称（右侧）
    for cat_name, (y_center, color) in [(k, (category_y_centers[k], cat_colors[k])) 
                                        for k in category_y_centers.keys()]:
        
        # 获取此类别的索引范围用于括号高度
        indices = categories[cat_name]
        height = len(indices) * 0.8
        
        # 绘制花括号 (使用弧线)
        bracket_x = 3.2
        
        # 简化的大括号：使用花括号字体或绘制线条
        # 绘制垂直线（括号主干）
        half_h = height / 2
        ax_c.plot([bracket_x, bracket_x], [y_center - half_h + 0.2, y_center + half_h - 0.2], 
                 color=color, linewidth=2, zorder=3)
        
        # 顶部钩子
        ax_c.plot([bracket_x, bracket_x + 0.15], [y_center - half_h + 0.2, y_center - half_h], 
                 color=color, linewidth=1.5)
        # 底部钩子  
        ax_c.plot([bracket_x, bracket_x + 0.15], [y_center + half_h - 0.2, y_center + half_h], 
                 color=color, linewidth=1.5)
        
        # 类别名称（缩写）放在括号右侧
        short_cat = cat_name.replace('Properties', 'Prop').replace('Dimension', 'Dim')
        ax_c.text(bracket_x + 0.25, y_center, short_cat, fontsize=6, 
                 va='center', ha='left', color=color, fontweight='bold',
                 rotation=0,
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                          edgecolor=color, alpha=0.9, linewidth=1))
    
    ax_c.set_title('(c) Violation Taxonomy\n(20 Categories)', fontsize=8, fontweight='bold', pad=4, loc='center')

    # 右侧统计文字（调整位置到图外）
    all_affected = len(all_mutants)
    stats_text = f"Collision: {all_affected}/{total_mutants}\({all_affected/total_mutants*100:.1f}%)"
    for i, s in enumerate(group_stats):
        stats_text += f"\nG{i+1}: {s['mutant_count']} muts, {s['diff_dimensions']}D diff"
    
    # 将统计信息放在图下方
    fig.text(0.5, 0.04, stats_text, fontsize=7, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.4))

    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.98)
    test_path = os.path.abspath(f'{output_filename}_collision_compact.png')    
    plt.savefig(test_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"文件已生成: {os.path.exists(test_path)}")
    
    plt.show()
    return fig
#endregion

#region PCA和t-SNE降维显示
def plot_tsne_comparison(violation_map, kill_matrix, mutant_ids, labels,title=''):
    """
    t-SNE 可视化行为指纹 vs Kill Matrix
    
    参数:
    - violation_map: {mutant_id: behavior_vector}
    - kill_matrix: {mutant_id: kill_vector}
    - mutant_ids: 变异体 ID 列表
    - labels: 聚类标签，用于着色
    """
    # 构建矩阵
    X_violation = np.array([violation_map[mid] for mid in mutant_ids])
    X_kill = np.array([kill_matrix[mid] for mid in mutant_ids])

    # t-SNE 投影
    tsne_v = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    Xv_2d = tsne_v.fit_transform(X_violation)

    tsne_k = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    Xk_2d = tsne_k.fit_transform(X_kill)

    # 绘图: 行为指纹
    plt.figure(figsize=(8, 6))
    plt.scatter(Xv_2d[:, 0], Xv_2d[:, 1], c=labels, cmap='tab10', s=50)
    plt.title(f"{title}: t-SNE - Behavioral Fingerprints")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster Label")
    plt.grid(True)
    plt.show()

    # 绘图: Kill Matrix
    plt.figure(figsize=(8, 6))
    plt.scatter(Xk_2d[:, 0], Xk_2d[:, 1], c=labels, cmap='tab10', s=50)
    plt.title(f"{title}: t-SNE - Kill Matrix")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Cluster Label")
    plt.grid(True)
    plt.show()


def plot_pca_comparison(violation_map, kill_matrix, mutant_ids, labels):
    X_violation = np.array([violation_map[mid] for mid in mutant_ids])
    X_kill = np.array([kill_matrix[mid] for mid in mutant_ids])

    pca = PCA(n_components=2)
    
    Xv_2d = pca.fit_transform(X_violation)
    Xk_2d = pca.fit_transform(X_kill)

    plt.figure()
    plt.scatter(Xv_2d[:, 0], Xv_2d[:, 1], c=labels)
    plt.title("PCA - Behavioral Fingerprints")

    plt.figure()
    plt.scatter(Xk_2d[:, 0], Xk_2d[:, 1], c=labels)
    plt.title("PCA - Kill Matrix")
    
    plt.show()
#endregion
    
#region 聚类指标，统一计算
def compute_clustering_metrics(X, labels):
    results = {}
    
    try:
        results['silhouette'] = silhouette_score(X, labels)
    except:
        results['silhouette'] = -1

    try:
        results['db_index'] = davies_bouldin_score(X, labels)
    except:
        results['db_index'] = float('inf')

    try:
        results['ch_score'] = calinski_harabasz_score(X, labels)
    except:
        results['ch_score'] = 0

    return results
#endregion

#region 基线对比：killmatrix
def compare_with_kill_matrix(violation_map, kill_matrix, mutant_ids, n_clusters):
    
    X_violation = np.array([violation_map[mid] for mid in mutant_ids])
    X_kill = np.array([kill_matrix[mid] for mid in mutant_ids])

    kmeans_v = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_v = kmeans_v.fit_predict(X_violation)

    kmeans_k = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(X_kill)

    metrics_v = compute_clustering_metrics(X_violation, labels_v)
    metrics_k = compute_clustering_metrics(X_kill, labels_k)

    return metrics_v, metrics_k, labels_v, labels_k
#endregion

#region 簇内簇间分离比指标
def compute_cluster_distances(X, labels):
    """
    计算簇内/簇间距离（欧氏 + 余弦）
    """
    unique_labels = np.unique(labels)

    intra_euc = []
    intra_cos = []
    inter_euc = []
    inter_cos = []

    # === 簇内距离 ===
    for lab in unique_labels:
        cluster_points = X[labels == lab]
        if len(cluster_points) > 1:
            intra_euc.extend(pdist(cluster_points, metric='euclidean'))
            intra_cos.extend(pdist(cluster_points, metric='cosine'))

    # === 簇间距离 ===
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            ci = X[labels == unique_labels[i]]
            cj = X[labels == unique_labels[j]]
            
            inter_euc.extend(cdist(ci, cj, metric='euclidean').flatten())
            inter_cos.extend(cdist(ci, cj, metric='cosine').flatten())

    # === 平均值 ===
    intra_euc_mean = np.mean(intra_euc)
    intra_cos_mean = np.mean(intra_cos)
    inter_euc_mean = np.mean(inter_euc)
    inter_cos_mean = np.mean(inter_cos)

    # === 分离比 ===
    ratio_euc = inter_euc_mean / intra_euc_mean if intra_euc_mean > 0 else 0
    ratio_cos = inter_cos_mean / intra_cos_mean if intra_cos_mean > 0 else 0

    return {
        'intra_euclidean': intra_euc_mean,
        'inter_euclidean': inter_euc_mean,
        'ratio_euclidean': ratio_euc,
        'intra_cosine': intra_cos_mean,
        'inter_cosine': inter_cos_mean,
        'ratio_cosine': ratio_cos
    }
#endregion


# =====================================================
# CLEAN PIPELINE (论文一致版本)
# =====================================================

def behavior_aware_reduction(
    kill_matrix,
    violation_map,
    ms_per_mutant,
    behavior_types,
    budget_ratio=0.2,
    mode="violation",  # "violation" | "random"
    random_state=42
):
    np.random.seed(random_state)

    mutant_ids = list(violation_map.keys())
    n_mutants = len(mutant_ids)
    budget = max(1, int(n_mutants * budget_ratio))

    # =====================================================
    # Stage 1: Coarse Clustering via KILL MATRIX
    # =====================================================
    X_kill = np.array([kill_matrix[mid] for mid in mutant_ids])

    max_k = min(10, max(2, n_mutants // 2))
    best_k, best_score = 2, -1

    for k in range(2, max_k + 1):
        if k >= n_mutants:
            break
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_kill)
            if len(set(labels)) > 1:
                score = silhouette_score(X_kill, labels)
                if score > best_score:
                    best_k, best_score = k, score
        except:
            continue

    # fallback（防止聚类失败）
    if best_score == -1:
        best_k = min(3, n_mutants)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_kill)

    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_labels):
        clusters[cid].append(mutant_ids[i])

    # =====================================================
    # Stage 2: Global Greedy Selection
    # =====================================================
    selected = []
    global_covered = set()
    remaining_budget = budget

    # violation 维度
    vec_dim = len(next(iter(violation_map.values())))
    all_violations = set(range(vec_dim))

    # 打乱 cluster 顺序（避免偏置）
    cluster_items = list(clusters.items())
    np.random.shuffle(cluster_items)

    for cid, members in cluster_items:
        if remaining_budget <= 0 or global_covered == all_violations:
            break

        # 按比例分配预算（但不强制至少1）
        cluster_budget = int(len(members) / n_mutants * budget)
        cluster_budget = min(cluster_budget, remaining_budget)

        if cluster_budget == 0:
            continue

        if mode == "violation":
            remaining = set(members)

            while remaining and cluster_budget > 0 and global_covered != all_violations:
                best_mid, best_gain = None, -1

                for mid in remaining:
                    vec = violation_map[mid]
                    idxs = set(np.where(vec > 0)[0])

                    # 🔥 全局 coverage（关键改动）
                    gain = len(idxs - global_covered)

                    if gain > best_gain:
                        best_gain = gain
                        best_mid = mid

                # 无增益则停止（允许 < budget）
                if best_mid is None or best_gain == 0:
                    break

                selected.append(best_mid)
                global_covered.update(
                    set(np.where(violation_map[best_mid] > 0)[0])
                )
                remaining.remove(best_mid)
                cluster_budget -= 1
                remaining_budget -= 1

                if remaining_budget <= 0:
                    break

        elif mode == "random":
            sampled = list(members)
            np.random.shuffle(sampled)

            for mid in sampled:
                if remaining_budget <= 0:
                    break
                selected.append(mid)
                remaining_budget -= 1

    # =====================================================
    # Metrics
    # =====================================================

    def get_violation_set(mids):
        vset = set()
        for mid in mids:
            idxs = np.where(violation_map[mid] > 0)[0]
            vset.update(idxs)
        return vset

    V_all = get_violation_set(mutant_ids)
    V_sel = get_violation_set(selected)

    VDR = len(V_sel) / len(V_all) * 100 if V_all else 0

    def is_killed(mid):
        return np.any(kill_matrix[mid] > 0)

    killed_all = set(mid for mid in mutant_ids if is_killed(mid))
    killed_selected = set(mid for mid in selected if is_killed(mid))

    MSR = (len(killed_selected) / len(killed_all) * 100) if killed_all else 0

    RR = 1 - len(selected) / n_mutants

    results = {
        "VDR": VDR,
        "MSR": MSR,
        "RR": RR,
        "selected": len(selected),
        "total": n_mutants,
        "clusters": best_k
    }

    return selected, results


# =====================================================
# TABLE III GENERATOR (论文直接用)
# =====================================================

def generate_tableIII(results_strict, results_relaxed):
    table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Priority Hierarchy Evaluation (RQ3)}}
\\begin{{tabular}}{{lccc}}
\\hline
Strategy & VDR (\\%) & MSR (\\%) & RR (\\%) \\
\\hline
Strict & {results_strict['VDR']:.1f} & {results_strict['MSR']:.1f} & {results_strict['RR']*100:.1f} \\
Relaxed & {results_relaxed['VDR']:.1f} & {results_relaxed['MSR']:.1f} & {results_relaxed['RR']*100:.1f} \\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    return table


# =====================================================
# EXPERIMENT RUNNER (可复现)
# =====================================================

def run_full_experiment(kill_matrix, violation_map, ms_per_mutant, behavior_types):

    print("\n[Running STRICT reduction]")
    sel_s, res_s = behavior_aware_reduction(
        kill_matrix, violation_map, ms_per_mutant, behavior_types,
        budget_ratio=0.1, mode="violation"
    )

    print("\n[Running RELAXED reduction]")
    sel_r, res_r = behavior_aware_reduction(
        kill_matrix, violation_map, ms_per_mutant, behavior_types,
        budget_ratio=0.2, mode="violation"
    )

    print("\n[Running ABLATION: random]")
    _, res_rand = behavior_aware_reduction(
        kill_matrix, violation_map, ms_per_mutant, behavior_types,
        budget_ratio=0.1, mode="random"
    )

    print("\n=== RESULTS ===")
    print("Strict:", res_s)
    print("Relaxed:", res_r)
    print("Random:", res_rand)

    print("\n=== TABLE III (LaTeX) ===")
    print(generate_tableIII(res_s, res_r))

    return res_s, res_r, res_rand

def compute_vdr(selected, violation_map):
    def get_vset(mids):
        v = set()
        for mid in mids:
            idxs = np.where(violation_map[mid] > 0)[0]
            v.update(idxs)
        return v
    
    all_ids = list(violation_map.keys())
    V_all = get_vset(all_ids)
    V_sel = get_vset(selected)
    
    return len(V_sel) / len(V_all) * 100 if V_all else 0

def kill_based_selection(kill_matrix, mutant_ids, budget_ratio):
    n = len(mutant_ids)
    budget = max(1, int(n * budget_ratio))
    
    X = np.array([kill_matrix[mid] for mid in mutant_ids])
    
    k = min(5, n)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    
    clusters = {}
    for i, cid in enumerate(labels):
        clusters.setdefault(cid, []).append(mutant_ids[i])
    
    selected = []
    
    for cid, members in clusters.items():
        cluster_budget = max(1, int(len(members)/n * budget))
        selected.extend(members[:cluster_budget])
    
    return selected[:budget]

def run_vdr_budget_curve(kill_matrix, violation_map, ms_per_mutant, behavior_types):
    
    budget_list = np.linspace(0.05, 0.5, 5)  # 5% → 50%
    
    vdr_violation = []
    vdr_random = []
    vdr_kill = []
    
    for b in budget_list:
        print(f"[Budget {b:.2f}]")
        
        # -----------------------------
        # 1. Violation-based（你的方法）
        # -----------------------------
        sel_v, _ = behavior_aware_reduction(
            kill_matrix, violation_map, ms_per_mutant, behavior_types,
            budget_ratio=b, mode="violation"
        )
        vdr_violation.append(compute_vdr(sel_v, violation_map))
        
        # -----------------------------
        # 2. Random baseline
        # -----------------------------
        sel_r, _ = behavior_aware_reduction(
            kill_matrix, violation_map, ms_per_mutant, behavior_types,
            budget_ratio=b, mode="random"
        )
        vdr_random.append(compute_vdr(sel_r, violation_map))
        
        # -----------------------------
        # 3. Kill-based baseline（重要！）
        # -----------------------------
        sel_k = kill_based_selection(kill_matrix, list(violation_map.keys()), b)
        vdr_kill.append(compute_vdr(sel_k, violation_map))
    
    # =============================
    # Plot
    # =============================
    plt.figure()
    plt.plot(budget_list, vdr_violation, marker='o', label='Violation-based')
    plt.plot(budget_list, vdr_random, marker='s', label='Random')
    plt.plot(budget_list, vdr_kill, marker='^', label='Kill-based')
    
    plt.xlabel("Budget Ratio")
    plt.ylabel("VDR (%)")
    plt.title("Behavior-Aware Reduction Under Different Budget")
    plt.legend()
    plt.grid()
    
    plt.show()
    
    return budget_list, vdr_violation, vdr_random, vdr_kill








