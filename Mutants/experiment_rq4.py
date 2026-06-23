import numpy as np
from collections import defaultdict
import math
import os
import matplotlib.pyplot as plt

def run_rq4_experiment_a(kill_matrix, violation_map, categories,
                                n_fine=20, survival_rate_threshold=0.90,
                                debug=True):
    """
    RQ4 Experiment A: CI Interception Rate (修正版)
    
    支持两种 violation_map 格式自动识别:
      - 20维细类频次向量: 按 categories 聚合到4层
      - 4维层向量: 直接使用
    
    Stage 1 采用"存活率阈值"定义功能测试放行:
      - 变异体级别: 存活率 >= survival_rate_threshold 视为"功能测试通过"
      - 用例对级别: kill=0 但 violation>0 视为"单用例漏检"
    """
    names = list(kill_matrix.keys())
    M = len(names)
    layer_names = list(categories.keys())
    
    # ---------- 0. 数据格式自检 ----------
    sample_vm = np.asarray(violation_map[names[0]], dtype=float)
    actual_dim = sample_vm.shape[0] if sample_vm.ndim == 1 else sample_vm.shape[1]
    
    if debug:
        print(f"[DEBUG] 变异体总数 M = {M}")
        print(f"[DEBUG] violation_map 样本维度 = {actual_dim} (期望 {n_fine} 或 4)")
        print(f"[DEBUG] kill_matrix 样本长度 = {len(kill_matrix[names[0]])}")
    
    # ---------- 1. 预计算四层指纹 ----------
    fp4 = {}  # name -> (4,) 层触发频次
    for n in names:
        vm = np.asarray(violation_map[n], dtype=float)
        
        if vm.shape == (4,):
            # 已经是4维层向量
            fp4[n] = vm
        elif vm.shape == (n_fine,):
            # 20维细类向量，需要聚合
            layer_counts = np.zeros(4)
            for i, indices in enumerate(categories.values()):
                valid_idx = [idx for idx in indices if 0 <= idx < n_fine]
                layer_counts[i] = vm[valid_idx].sum()
            fp4[n] = layer_counts
        else:
            raise ValueError(
                f"变异体 {n}: violation_map 维度 {vm.shape} 既不是 ({n_fine},) 也不是 (4,)"
            )
    
    if debug:
        print(f"[DEBUG] fp4 样本 (前3个):")
        for n in names[:3]:
            print(f"  {n}: {fp4[n]} -> dominant: {layer_names[int(np.argmax(fp4[n]))]}")
    
    # ---------- 2. 计算存活率 ----------
    survival_rates = {}
    for n in names:
        km = np.asarray(kill_matrix[n])
        # 自动推断 kill 语义: 0=存活/通过, 1=杀死/失败
        # 支持浮点数 (如 0.0/1.0) 和整数
        sr = np.mean(km == 0)  # 存活率 = 存活用例占比
        survival_rates[n] = float(sr)
    
    # 存活率分布
    sr_vals = list(survival_rates.values())
    if debug:
        print(f"[DEBUG] 存活率范围: [{min(sr_vals):.3f}, {max(sr_vals):.3f}]")
        print(f"[DEBUG] 存活率=1.0 (全通过) 的变异体数: {sum(1 for v in sr_vals if v == 1.0)}")
        print(f"[DEBUG] 存活率=0.0 (全失败) 的变异体数: {sum(1 for v in sr_vals if v == 0.0)}")
        print(f"[DEBUG] 存活率 >= {survival_rate_threshold} 的变异体数: "
              f"{sum(1 for v in sr_vals if v >= survival_rate_threshold)}")
    
    # ---------- 3. Stage 1: 功能测试放行 (变异体级别) ----------
    stage1_passed = [n for n in names if survival_rates[n] >= survival_rate_threshold]
    stage1_failed = [n for n in names if survival_rates[n] < survival_rate_threshold]
    N_passed = len(stage1_passed)
    N_failed = len(stage1_failed)
    
    if debug:
        print(f"\n[DEBUG] Stage 1 (功能测试):")
        print(f"  通过 (存活率 >= {survival_rate_threshold}): {N_passed}")
        print(f"  失败 (存活率 < {survival_rate_threshold}): {N_failed}")
    
    if N_passed == 0:
        print("[WARNING] Stage 1 无通过变异体！尝试降低 survival_rate_threshold 或检查 kill_matrix 语义。")
        # 降级: 只要有任何存活用例就算"通过"
        stage1_passed = [n for n in names if survival_rates[n] > 0]
        N_passed = len(stage1_passed)
        print(f"[FALLBACK] 使用存活率 > 0 作为通过标准: {N_passed} 个")
    
    # ---------- 4. Stage 2: HVAR 四层违规审计 (变异体级别) ----------
    intercepted = []      # 通过但触发违规
    clean_pass = []       # 通过且无违规
    layer_intercepted = defaultdict(list)
    
    for n in stage1_passed:
        fp = fp4[n]
        if np.any(fp > 0):
            intercepted.append(n)
            for l in range(4):
                if fp[l] > 0:
                    layer_intercepted[l].append(n)
        else:
            clean_pass.append(n)
    
    N_intercepted = len(intercepted)
    N_clean = len(clean_pass)
    
    # ---------- 5. 用例对级别分析 (Pair-level) ----------
    # 对每个 (mutant, test_case) 对，检查 kill=0 但 violation>0
    pair_level = {
        'total_pairs': 0,
        'km_passed_pairs': 0,           # kill=0
        'hvar_intercepted_pairs': 0,   # kill=0 & violation>0
        'true_clean_pairs': 0,         # kill=0 & violation=0
    }
    
    # 用例对级别的层拦截统计
    pair_layer_intercepted = defaultdict(int)
    
    for n in stage1_passed:
        km = np.asarray(kill_matrix[n])
        fp = fp4[n]
        # 这里假设 violation_map 是变异体级别的聚合
        # 用例对级别需要原始逐用例的违规判定，但现有数据是聚合的
        # 我们用 fp4 的"是否在该层有任何触发"作为代理
        # 更精确的做法需要逐用例 violation 矩阵，这里做变异体级聚合近似
        
        # 如果用户有逐用例的 violation 矩阵 (M x 4)，可以替换此处
        n_tests = len(km)
        for t in range(n_tests):
            pair_level['total_pairs'] += 1
            if km[t] == 0:  # 该用例存活
                pair_level['km_passed_pairs'] += 1
                # 由于我们只有聚合指纹，无法精确到单个用例的违规
                # 近似: 如果变异体在该层有触发，假设部分通过用例触发了它
                # 精确实现需要 violation_matrix[M, T, 4] 的三维数据
    
    # 注: 若需精确用例对级别，请提供 violation_matrix[mutant][test_case] -> (4,) 的逐用例违规标记
    # 以下为基于聚合指纹的变异体级指标 (已足够支撑论文)
    
    # ---------- 6. 核心指标计算 ----------
    IR = N_intercepted / N_passed if N_passed > 0 else 0.0  # Interception Rate
    
    LIR = {}
    for l in range(4):
        LIR[layer_names[l]] = (
            len(layer_intercepted[l]) / N_passed if N_passed > 0 else 0.0
        )
    
    # 主导违规分布 (在 Intercepted 集合内)
    dominant_dist = defaultdict(int)
    for n in intercepted:
        dom_l = int(np.argmax(fp4[n]))
        dominant_dist[layer_names[dom_l]] += 1
    
    CPR = N_clean / N_passed if N_passed > 0 else 0.0  # Clean Pass Rate
    
    # ---------- 7. 对照组: Stage 1 失败变异体的违规谱 ----------
    failed_layer_trigger = defaultdict(int)
    failed_dominant_dist = defaultdict(int)
    for n in stage1_failed:
        fp = fp4[n]
        for l in range(4):
            if fp[l] > 0:
                failed_layer_trigger[layer_names[l]] += 1
        dom_l = int(np.argmax(fp))
        failed_dominant_dist[layer_names[dom_l]] += 1
    
    return {
        'operator_summary': {
            'total_mutants': M,
            'stage1_passed': N_passed,
            'stage1_failed': N_failed,
            'stage2_intercepted': N_intercepted,
            'stage2_clean_pass': N_clean,
        },
        'core_metrics': {
            'Interception_Rate_IR': round(float(IR), 4),
            'Clean_Pass_Rate_CPR': round(float(CPR), 4),
            'Layer_Interception_Rate_LIR': {
                k: round(float(v), 4) for k, v in LIR.items()
            },
        },
        'intercepted_analysis': {
            'dominant_violation_distribution': dict(dominant_dist),
            'intercepted_ids': intercepted,
            'clean_pass_ids': clean_pass,
        },
        'survival_rate_distribution': {
            'min': round(min(sr_vals), 3),
            'max': round(max(sr_vals), 3),
            'mean': round(np.mean(sr_vals), 3),
            'median': round(np.median(sr_vals), 3),
        },
        'control_group_failed': {
            'failed_count': N_failed,
            'failed_layer_trigger_distribution': dict(failed_layer_trigger),
            'failed_dominant_distribution': dict(failed_dominant_dist),
        }
    }


import numpy as np
from collections import defaultdict
import math

def run_rq4_experiment_b(intercepted_mutants, violation_map, categories, n_fine=20, n_bins=6):
    """
    RQ4 Experiment B v2: Diagnostic Richness (修正版)
    
    核心修正：
    1. DSC 基于指纹唯一模式数，而非主导层种类数
    2. DE 基于指纹空间分布熵（分箱离散化），而非主导层标签熵
    3. 新增 MLCR：多层耦合率
    """
    layer_names = list(categories.keys())
    
    # 预计算四层聚合指纹
    fp4 = {}
    for n in intercepted_mutants:
        fine_vec = np.asarray(violation_map[n], dtype=float)
        layer_counts = np.zeros(4)
        for i, indices in enumerate(categories.values()):
            valid_idx = [idx for idx in indices if 0 <= idx < n_fine]
            layer_counts[i] = fine_vec[valid_idx].sum()
        fp4[n] = layer_counts
    
    # 构建指纹矩阵 (n, 4)
    X = np.array([fp4[n] for n in intercepted_mutants])
    n_samples = len(intercepted_mutants)
    
    # ---------- 1. DSC: Diagnostic State Count ----------
    # Kill-Matrix: 对 Passed 集合只有 1 个状态
    dsc_km = 1
    
    # Fingerprint v2: 唯一指纹向量数（round到整数，或基于聚类）
    # 方法A: 严格唯一向量
    unique_fp_strict = len(set(tuple(v.round(0).astype(int)) for v in X))
    # 方法B: 基于分箱的离散模式数（更稳健）
    bins = np.linspace(X.min(), X.max(), n_bins + 1)
    digitized = np.digitize(X, bins)
    unique_fp_binned = len(set(tuple(v) for v in digitized))
    
    dsc_fp_strict = unique_fp_strict
    dsc_fp_binned = unique_fp_binned
    
    # ---------- 2. MLCR: Multi-Layer Coupling Rate ----------
    # 触发 >= 2 层的变异体占比
    triggered_layers_per_mutant = np.sum(X > 0, axis=1)
    mlcr = np.mean(triggered_layers_per_mutant >= 2)
    
    # ---------- 3. DE v2: Fingerprint Space Entropy ----------
    # 基于分箱离散化的联合分布熵
    # 每维独立分箱，计算联合直方图
    hist, edges = np.histogramdd(X, bins=[n_bins]*4)
    hist_flat = hist.flatten()
    hist_prob = hist_flat[hist_flat > 0] / hist_flat.sum()
    
    de_fp = -np.sum(hist_prob * np.log2(hist_prob))
    # 最大可能熵（均匀分布）
    de_max = np.log2(n_bins ** 4)
    de_normalized = de_fp / de_max if de_max > 0 else 0
    
    # Kill-Matrix 的熵 = 0（只有1个状态）
    de_km = 0.0
    entropy_gain = de_fp - de_km
    
    # ---------- 4. 修复导向性 (FMR) ----------
    # 对每条指纹，基于主导层映射修复策略
    fix_strategy = {
        0: 'Numerical guards (epsilon, max-trick, precision upgrade)',
        1: 'Recalibrate statistics (mean/variance correction)',
        2: 'Verify normalization axioms (sum-to-one, non-negativity)',
        3: 'Check structural constraints (dimension, monotonicity)'
    }
    
    # FMR_FP: 100%（每条指纹都有明确主导层和修复建议）
    fmr_fp = 100.0
    
    # FMR_KM: 0%（Kill-Matrix对Passed变异体无修复建议）
    fmr_km = 0.0
    
    # 修复策略分布（基于主导层，用于辅助分析）
    dominant_layers = [int(np.argmax(fp4[n])) for n in intercepted_mutants]
    fix_dist = defaultdict(int)
    for l in dominant_layers:
        fix_dist[fix_strategy[l]] += 1
    
    # ---------- 5. 典型案例（展示同主导层、异指纹配方）----------
    # 找主导层相同但指纹差异最大的配对
    case_reports = []
    
    # 按主导层分组
    groups = defaultdict(list)
    for n in intercepted_mutants:
        dom = int(np.argmax(fp4[n]))
        groups[dom].append(n)
    
    # 在最大的组内找差异最大的案例
    for dom_l, members in groups.items():
        if len(members) < 2:
            continue
        
        # 找 L1 距离最大的配对
        best_pair = None
        best_dist = -1
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                d = np.sum(np.abs(fp4[members[i]] - fp4[members[j]]))
                if d > best_dist:
                    best_dist = d
                    best_pair = (members[i], members[j])
        
        if best_pair:
            m1, m2 = best_pair
            case_reports.append({
                'mutant_1': m1,
                'mutant_2': m2,
                'dominant_layer': layer_names[dom_l],
                'fp_m1': {layer_names[i]: int(fp4[m1][i]) for i in range(4)},
                'fp_m2': {layer_names[i]: int(fp4[m2][i]) for i in range(4)},
                'l1_distance': int(best_dist),
                'km_diagnosis': 'Survived (all tests passed, no anomaly detected)',
                'fp_insight_m1': f"Numerical Stability={int(fp4[m1][0])}, Distributional Axiom={int(fp4[m1][2])}",
                'fp_insight_m2': f"Numerical Stability={int(fp4[m2][0])}, Distributional Axiom={int(fp4[m2][2])}",
                'suggested_fix': fix_strategy[dom_l]
            })
        
        if len(case_reports) >= 3:
            break
    
    return {
        'sample_size': n_samples,
        'granularity': {
            'DSC_KM': dsc_km,
            'DSC_FP_strict': dsc_fp_strict,      # 唯一整数指纹数
            'DSC_FP_binned': dsc_fp_binned,      # 分箱离散模式数
            'MLCR': round(float(mlcr), 3),       # 多层耦合率
            'avg_triggered_layers': round(float(np.mean(triggered_layers_per_mutant)), 2)
        },
        'fix_directness': {
            'FMR_KM': fmr_km,
            'FMR_FP': fmr_fp,
            'strategy_distribution': dict(fix_dist)
        },
        'diagnostic_entropy': {
            'DE_KM': round(float(de_km), 3),
            'DE_FP_raw': round(float(de_fp), 3),
            'DE_FP_normalized': round(float(de_normalized), 3),
            'DE_max_possible': round(float(de_max), 3),
            'entropy_gain': round(float(entropy_gain), 3)
        },
        'case_reports': case_reports
    }


import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# ============================================================
# 统一全局样式（与 RQ1 完全一致）
# ============================================================
STYLE = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'CMU Serif'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.grid.axis': 'y',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}
for k, v in STYLE.items():
    plt.rcParams[k] = v

# 统一配色（确保全部定义）
C_PRIMARY = '#2E5C8A'
C_SECONDARY = '#C0554F'
C_GREEN = '#5A8A6A'
C_TEXT = '#2B2D42'
C_GRAY = '#6B7280'
C_LIGHT = '#F5F5F5'  # ← 这里必须定义

# ============================================================
# 图1：存活组四层触发率
# ============================================================
def plot_rq4_fig1():
    output_dir = "RQ4"
    os.makedirs(output_dir, exist_ok=True)

    operators = ['Softmax', 'LayerNorm', 'RBF Kernel']
    layers = ['L1\n(Numerical\nStability)', 'L2\n(Statistical\nMoments)',
              'L3\n(Distributional\nAxioms)', 'L4\n(Structural\nInvariants)']
    ltr_data = {
        'Softmax': [100, 100, 100, 100],
        'LayerNorm': [100, 100, 100, 100],
        'RBF Kernel': [100, 89, 72, 15]
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(layers))
    width = 0.22
    colors = [C_PRIMARY, C_SECONDARY, C_GREEN]

    for i, op in enumerate(operators):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, ltr_data[op], width, label=op,
                      color=colors[i], edgecolor='white', linewidth=0.6, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color=colors[i])

    ax.set_ylabel('Layer Trigger Rate (%)', fontsize=11)
    ax.set_title('RQ4: Four-Layer Violation Trigger Rates within Survived Group',
                 fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(title='Primitive', loc='upper left', bbox_to_anchor=(1, 1),
              frameon=False, fontsize=9, borderaxespad=0)

    ax.axhline(y=100, color=C_GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(1.7, 101, '100% threshold', fontsize=8, color=C_GRAY, ha='right')

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{output_dir}/rq4_fig1_layer_trigger_rates.png", dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print("图1已保存")

# ============================================================
# 图2：存活组 vs 杀死组 质心对比
# ============================================================
def plot_rq4_fig2():
    output_dir = "RQ4"
    os.makedirs(output_dir, exist_ok=True)

    operators = ['Softmax', 'LayerNorm', 'RBF Kernel']
    layers = ['L1', 'L2', 'L3', 'L4']
    survived_centroids = {
        'Softmax': [231, 162, 367, 30],
        'LayerNorm': [60, 209, 312, 22],
        'RBF Kernel': [342, 15, 28, 8]
    }
    killed_centroids = {
        'Softmax': [89, 45, 120, 15],
        'LayerNorm': [25, 78, 95, 10],
        'RBF Kernel': [410, 22, 35, 12]
    }
    icd_values = {'Softmax': 493, 'LayerNorm': 387, 'RBF Kernel': 521}

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('RQ4: Normalized Centroid Profiles of Survived vs. Killed Groups',
                 fontsize=13, fontweight='bold', color=C_TEXT, y=0.98)

    x_layer = np.arange(len(layers))

    # ========== 上行：绝对值柱状图（保留，用于展示数值差异） ==========
    width_bar = 0.32
    for j, op in enumerate(operators):
        ax = fig.add_subplot(2, 3, j + 1)
        s_vals = survived_centroids[op]
        k_vals = killed_centroids[op]

        ax.bar(x_layer - width_bar / 2, s_vals, width_bar, label='Survived',
               color=C_PRIMARY, edgecolor='white', linewidth=0.6, alpha=0.9)
        ax.bar(x_layer + width_bar / 2, k_vals, width_bar, label='Killed',
               color=C_SECONDARY, edgecolor='white', linewidth=0.6, alpha=0.85)

        ax.set_title(f'{op}  (ICD = {icd_values[op]})', fontsize=11, fontweight='bold', color=C_TEXT)
        ax.set_xticks(x_layer)
        ax.set_xticklabels(layers, fontsize=9)
        if j == 0:
            ax.set_ylabel('Violation Trigger Count', fontsize=11)
            ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0, 450)

    # ========== 下行：归一化折线图（展示轮廓形状） ==========
    for j, op in enumerate(operators):
        ax = fig.add_subplot(2, 3, j + 4)
        s_vals = np.array(survived_centroids[op])
        k_vals = np.array(killed_centroids[op])

        # 各自归一化到最大值
        s_norm = s_vals / s_vals.max()
        k_norm = k_vals / k_vals.max()

        # Survived：实心圆点 + 实线
        ax.plot(x_layer, s_norm, 'o-', color=C_PRIMARY, linewidth=2.5, markersize=8,
                markerfacecolor=C_PRIMARY, markeredgecolor='white', markeredgewidth=1.5,
                label='Survived', zorder=3)

        # Killed：空心方块 + 虚线
        ax.plot(x_layer, k_norm, 's--', color=C_SECONDARY, linewidth=2.5, markersize=8,
                markerfacecolor='white', markeredgecolor=C_SECONDARY, markeredgewidth=1.5,
                label='Killed', zorder=3)

        # 添加淡色水平参考线，帮助判断主导层
        ax.axhline(y=1.0, color=C_GRAY, linestyle=':', linewidth=0.6, alpha=0.4)
        ax.axhline(y=0.5, color=C_GRAY, linestyle=':', linewidth=0.6, alpha=0.3)

        ax.set_title(f'{op} Normalized Profile', fontsize=11, fontweight='bold', color=C_TEXT, pad=8)
        ax.set_xticks(x_layer)
        ax.set_xticklabels(layers, fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=9)

        if j == 0:
            ax.set_ylabel('Normalized Trigger Count', fontsize=11)
            ax.legend(frameon=False, fontsize=9, loc='upper right')

        # 极淡的 x 轴网格
        ax.grid(axis='x', linestyle=':', alpha=0.2, zorder=0)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
    plt.show()
    plt.savefig(f"{output_dir}/rq4_fig2_centroid_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图2已保存")
# ============================================================
# 图3：SCR 与 DLD 组合图
# ============================================================
def plot_rq4_fig3():
    output_dir = "RQ4"
    os.makedirs(output_dir, exist_ok=True)

    operators = ['Softmax', 'LayerNorm', 'RBF Kernel']
    scr_values = {'Softmax': 0.125, 'LayerNorm': 0.250, 'RBF Kernel': 0.500}
    dld_data = {
        'Softmax': [27, 13, 60],
        'LayerNorm': [12, 32, 56],
        'RBF Kernel': [89, 5, 6]
    }
    dld_labels = ['L1', 'L2', 'L3']

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    x_op = np.arange(len(operators))
    width = 0.22

    colors_dld = [C_PRIMARY, C_SECONDARY, C_GREEN]
    for i, label in enumerate(dld_labels):
        vals = [dld_data[op][i] for op in operators]
        offset = width * (i - 1)
        ax1.bar(x_op + offset, vals, width, label=label,
                color=colors_dld[i], edgecolor='white', linewidth=0.6, alpha=0.9)

    ax1.set_ylabel('Dominant Layer Distribution (%)', fontsize=11)
    ax1.set_title('RQ4: State Compression Ratio and Dominant-Layer Distribution',
                  fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax1.set_xticks(x_op)
    ax1.set_xticklabels(operators, fontsize=9.5)
    ax1.set_ylim(0, 110)
    ax1.legend(title='Dominant Layer', loc='upper left', frameon=False, fontsize=9)

    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    scr_list = [scr_values[op] for op in operators]
    ax2.plot(x_op, scr_list, 'D-', color=C_SECONDARY, linewidth=2, markersize=7,
             markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=C_SECONDARY, zorder=5)
    for i, v in enumerate(scr_list):
        ax2.annotate(f'{v:.3f}', (x_op[i], v), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color=C_SECONDARY, fontweight='bold')
    ax2.set_ylabel('State Compression Ratio (SCR)', fontsize=11, color=C_SECONDARY)
    ax2.set_ylim(0, 0.6)
    ax2.tick_params(axis='y', labelcolor=C_SECONDARY)

    lines1, labels1 = ax1.get_legend_handles_labels()
    line2 = Line2D([0], [0], color=C_SECONDARY, marker='D', linestyle='-', markerfacecolor='white',
                   markeredgewidth=1.8, markersize=7, label='SCR')
    ax1.legend(handles=lines1 + [line2], loc='upper left', frameon=False, fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{output_dir}/rq4_fig3_scr_and_dld.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图3已保存")


# ============================================================
# 图4：典型案例 P49 vs P52
# ============================================================
def plot_rq4_fig4():
    output_dir = "RQ4"
    os.makedirs(output_dir, exist_ok=True)

    layers = ['L1\n(Numerical\nStability)', 'L2\n(Statistical\nMoments)',
              'L3\n(Distributional\nAxioms)', 'L4\n(Structural\nInvariants)']
    p49 = [256, 162, 343, 30]
    p52 = [228, 162, 364, 30]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_case = np.arange(len(layers))
    width = 0.32

    bars1 = ax.bar(x_case - width / 2, p49, width, label='P49',
                   color=C_PRIMARY, edgecolor='white', linewidth=0.6, alpha=0.9)
    bars2 = ax.bar(x_case + width / 2, p52, width, label='P52',
                   color=C_SECONDARY, edgecolor='white', linewidth=0.6, alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.annotate('', xy=(0.15, 252), xytext=(0.85, 252),
                arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=1.8))
    ax.text(0.5, 255, 'L1 dist = 49', ha='center', fontsize=9, color=C_GREEN, fontweight='bold')

    ax.set_ylabel('Violation Trigger Count', fontsize=11)
    ax.set_title('RQ4: Case Study — Survived Softmax Mutants P49 vs. P52',
                 fontsize=13, fontweight='bold', color=C_TEXT, pad=12)
    ax.set_xticks(x_case)
    ax.set_xticklabels(layers, fontsize=9)
    ax.legend(title='Mutant', loc='upper right', frameon=False, fontsize=9)
    ax.set_ylim(0, 420)

    ax.text(2.5, 380, 'Dominant Layer: L3', fontsize=10, ha='center', color=C_SECONDARY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LIGHT, edgecolor=C_SECONDARY, linewidth=0.8))

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{output_dir}/rq4_fig4_case_p49_p52.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图4已保存")

def experiment_qr4_plot():
    plot_rq4_fig1()
    plot_rq4_fig2()
    plot_rq4_fig3()
    plot_rq4_fig4()






