import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ============================================
# 实验二数据（基于你最新运行结果硬编码）
# ============================================
def plotPerturb_test_softmax():

    layers = ['L1\\nNumerical\\nStability', 'L2\\nStatistical\\nMoments', 
            'L3\\Distributional\\nAxiom', 'L4\\nStructural\\nInvariants']
    layers_short = ['L1', 'L2', 'L3', 'L4']

    strategies = {
        'Clean':                [0.1850, 0.5250, 0.2000, 0.0350],
        'Gaussian Noise':       [0.1700, 0.5300, 0.2150, 0.0100],
        'Extreme Boundary':     [0.3950, 0.6950, 0.3950, 0.0300],
        'Adversarial Shift':    [0.1700, 0.5250, 0.2100, 0.0150],
        'Precision Degradation':[0.1850, 0.5250, 0.2000, 0.0350],
    }

    # 统计检验数据: (delta_vtr, cohens_d, pvalue, absorbed)
    # absorbed=True 表示饱和吸收, False 表示显著穿透
    stats_data = {
        'Gaussian Noise': {
            'L1': (-0.0150, -0.1231, 0.6767, True),
            'L2': ( 0.0050,  0.0707, 0.8884, True),
            'L3': ( 0.0150,  0.1231, 0.6767, True),
            'L4': (-0.0250, -0.1597, 0.4913, True),
        },
        'Extreme Boundary': {
            'L1': ( 0.2100,  0.4858, 0.0000, False),
            'L2': ( 0.1700,  0.4514, 0.0000, False),
            'L3': ( 0.1950,  0.4760, 0.0000, False),
            'L4': (-0.0050, -0.0707, 0.8884, True),
        },
        'Adversarial Shift': {
            'L1': (-0.0150, -0.1231, 0.6767, True),
            'L2': ( 0.0000,  0.0000, 1.0000, True),
            'L3': ( 0.0100,  0.1003, 0.7800, True),
            'L4': (-0.0200, -0.1425, 0.5801, True),
        },
        'Precision Degradation': {
            'L1': ( 0.0000,  0.0000, 1.0000, True),
            'L2': ( 0.0000,  0.0000, 1.0000, True),
            'L3': ( 0.0000,  0.0000, 1.0000, True),
            'L4': ( 0.0000,  0.0000, 1.0000, True),
        },
    }

    # 细粒度违规 Top 数据 (VTR)
    fine_grained = {
        'Clean': {
            'numerical_underflow': 0.485, 'degenerate_distribution': 0.120,
            'low_entropy_sharp': 0.175, 'monotonicity_violation': 0.215,
            'topk_inconsistency': 0.200, 'probability_sum_violation': 0.020,
            'probability_contraction': 0.020, 'gradient_saturation': 0.185,
            'broadcasting_error': 0.015, 'cross_class_violation': 0.020,
        },
        'Extreme Boundary': {
            'numerical_underflow': 0.575, 'degenerate_distribution': 0.090,
            'low_entropy_sharp': 0.155, 'monotonicity_violation': 0.395,
            'topk_inconsistency': 0.375, 'probability_sum_violation': 0.190,
            'probability_contraction': 0.015, 'gradient_saturation': 0.270,
            'nan': 0.175, 'broadcasting_error': 0.030,
        }
    }

    # ============================================
    # 图1: 四层违规触发率 VTR 分组柱状图
    # ============================================
    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = np.arange(len(layers_short))
    width = 0.15
    colors = ['#2C3E50', '#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
    hatches = ['', '', '///', '', 'xxx']

    for i, (name, vals) in enumerate(strategies.items()):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i], 
                    edgecolor='black', linewidth=0.5, hatch=hatches[i], alpha=0.9)
        
        # 在柱顶标注数值
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Violation Trigger Rate (VTR)', fontsize=11)
    ax.set_xlabel('Violation Layer', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(layers_short, fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.legend(loc='upper left', frameon=True, fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(3.6, 0.052, '5% baseline', fontsize=7, color='gray', ha='right')

    # 添加显著性标记：Extreme Boundary 在 L1-L3 上标 *
    for j, layer in enumerate(layers_short):
        extreme_val = strategies['Extreme Boundary'][j]
        clean_val = strategies['Clean'][j]
        if stats_data['Extreme Boundary'][layer][3] == False:  # 显著穿透
            y_pos = extreme_val + 0.03
            ax.plot([j + width, j + width], [extreme_val, y_pos], 'k-', linewidth=1)
            ax.text(j + width, y_pos + 0.01, '*', ha='center', fontsize=14, color='#E74C3C')

    plt.tight_layout()
    plt.savefig('perterb_test_result/exp2_vtr_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('perterb_test_result/exp2_vtr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图1已保存: exp2_vtr_comparison.pdf / .png")

    # ============================================
    # 图2: 效应量热力图 (Cohen's d + p-value)
    # ============================================
    fig, ax = plt.subplots(figsize=(7, 4.5))

    perturb_names = ['Gaussian\\nNoise', 'Extreme\\nBoundary', 'Adversarial\\nShift', 'Precision\\nDegradation']
    perturb_keys = ['Gaussian Noise', 'Extreme Boundary', 'Adversarial Shift', 'Precision Degradation']

    # 构建 Cohen's d 矩阵
    d_matrix = np.zeros((4, 4))
    p_matrix = np.zeros((4, 4))
    absorbed_matrix = np.zeros((4, 4), dtype=bool)

    for i, pk in enumerate(perturb_keys):
        for j, layer in enumerate(layers_short):
            d_matrix[i, j] = stats_data[pk][layer][1]
            p_matrix[i, j] = stats_data[pk][layer][2]
            absorbed_matrix[i, j] = stats_data[pk][layer][3]

    # 自定义颜色映射：蓝色=饱和(负/小), 红色=穿透(正大)
    cmap = LinearSegmentedColormap.from_list('custom', ['#3498DB', '#FFFFFF', '#E74C3C'], N=256)
    im = ax.imshow(d_matrix, cmap=cmap, aspect='auto', vmin=-0.5, vmax=0.5)

    # 在单元格内标注
    for i in range(4):
        for j in range(4):
            d_val = d_matrix[i, j]
            p_val = p_matrix[i, j]
            absorbed = absorbed_matrix[i, j]
            
            text_color = 'white' if abs(d_val) > 0.3 else 'black'
            label = 'Absorbed' if absorbed else 'Penetrated'
            
            ax.text(j, i, f'd={d_val:.3f}\np={p_val:.3f}\n{label}', 
                    ha='center', va='center', fontsize=8, color=text_color, fontweight='bold')

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(layers_short, fontsize=10)
    ax.set_yticklabels(perturb_names, fontsize=10)
    ax.set_title('Statistical Saturation Test (Cohen\'s d & Wilcoxon p-value)', fontsize=12, pad=10)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cohen\'s d', fontsize=10)
    cbar.ax.axhline(y=0.2, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=-0.2, color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.5, 0.22, '±0.2 threshold', fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig('perterb_test_result/exp2_effect_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('perterb_test_result/exp2_effect_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图2已保存: exp2_effect_heatmap.pdf / .png")

    # ============================================
    # 图3: 细粒度违规触发谱 (Extreme vs Clean)
    # ============================================
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # 选取 Top 10 有区分度的违规类型
    viols = ['numerical_underflow', 'monotonicity_violation', 'topk_inconsistency',
            'gradient_saturation', 'probability_sum_violation', 'degenerate_distribution',
            'low_entropy_sharp', 'nan', 'broadcasting_error', 'probability_contraction']
    viols_labels = ['Num.Underflow', 'Monotonicity', 'Top-K Incon.', 
                    'Grad.Saturation', 'Prob.Sum Viol.', 'Degenerate',
                    'Low Entropy', 'NaN', 'Broadcast Err.', 'Prob.Contraction']

    x = np.arange(len(viols))
    width = 0.35

    clean_vals = [fine_grained['Clean'].get(v, 0) for v in viols]
    extreme_vals = [fine_grained['Extreme Boundary'].get(v, 0) for v in viols]

    bars1 = ax.bar(x - width/2, clean_vals, width, label='Clean', color='#2C3E50', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, extreme_vals, width, label='Extreme Boundary', color='#E74C3C', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('VTR', fontsize=11)
    ax.set_xlabel('Fine-Grained Violation Type', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(viols_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注仅在 Extreme 下激活的违规
    for i, (c, e) in enumerate(zip(clean_vals, extreme_vals)):
        if c == 0 and e > 0:
            ax.text(i + width/2, e + 0.02, 'Activated', ha='center', fontsize=7, 
                    color='#E74C3C', fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig('perterb_test_result/exp2_finegrained_spectrum.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('perterb_test_result/exp2_finegrained_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图3已保存: exp2_finegrained_spectrum.pdf / .png")

    # ============================================
    # 图4: 综合饱和吸收判定图 (Summary)
    # ============================================
    fig, ax = plt.subplots(figsize=(6, 4))

    # 构建判定矩阵：1=Absorbed, 0=Penetrated
    judgment = np.zeros((4, 4), dtype=int)
    for i, pk in enumerate(perturb_keys):
        for j, layer in enumerate(layers_short):
            judgment[i, j] = 1 if stats_data[pk][layer][3] else 0

    # 用方块图展示
    colors_judge = ['#E74C3C', '#2ECC71']  # 红=穿透, 绿=吸收
    cmap_judge = LinearSegmentedColormap.from_list('judge', colors_judge, N=2)
    im2 = ax.imshow(judgment, cmap=cmap_judge, aspect='auto')

    for i in range(4):
        for j in range(4):
            text = 'Absorbed' if judgment[i,j] == 1 else 'Penetrated'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, 
                    color='white', fontweight='bold')

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(layers_short, fontsize=10)
    ax.set_yticklabels(perturb_names, fontsize=10)
    ax.set_title('Saturation Absorption Verdict', fontsize=12, pad=10)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ECC71', edgecolor='black', label='Absorbed (Saturation)'),
                    Patch(facecolor='#E74C3C', edgecolor='black', label='Penetrated')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
            ncol=2, frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig('perterb_test_result/exp2_saturation_verdict.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('perterb_test_result/exp2_saturation_verdict.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图4已保存: exp2_saturation_verdict.pdf / .png")