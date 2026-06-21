import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from collections import defaultdict

def compute_cm_classes(kill_matrix, violation_map):
    keys = sorted(kill_matrix.keys())
    X = np.array([violation_map[k] for k in keys])
    sig_dict = {}
    for idx, k in enumerate(keys):
        sig_dict.setdefault(np.asarray(kill_matrix[k]).tobytes(), []).append(idx)
    groups = [c for c in sig_dict.values() if len(c) >= 2]
    total_pairs = mismatch_pairs = 0
    class_cm, class_sizes = [], []
    for grp in groups:
        tp = mp = 0
        for i, j in combinations(range(len(grp)), 2):
            tp += 1
            if not np.array_equal(X[grp[i]], X[grp[j]]):
                mp += 1
        class_cm.append(mp / tp if tp else 0.0)
        class_sizes.append(len(grp))
        total_pairs += tp
        mismatch_pairs += mp
    overall = mismatch_pairs / total_pairs if total_pairs else 0.0
    return overall, class_cm, class_sizes, len(groups)

def compute_cm_fp_strict(violation_map):
    keys = sorted(violation_map.keys())
    X = np.array([violation_map[k] for k in keys])
    sig_dict = {}
    for idx, k in enumerate(keys):
        sig_dict.setdefault(X[idx].tobytes(), []).append(idx)
    groups = [c for c in sig_dict.values() if len(c) >= 2]
    class_cm = [0.0] * len(groups)
    class_sizes = [len(g) for g in groups]
    return 0.0, class_cm, class_sizes, len(groups)

def _safe_kde(data, y_range, width=0.35):
    if len(data) < 3 or len(np.unique(data)) < 2:
        return None
    try:
        d = gaussian_kde(data)(y_range)
        return d / d.max() * width
    except:
        return None

def plot_cm_both(kill_matrix, violation_map):
    ovr_k, cm_k, sz_k, n_cls = compute_cm_classes(kill_matrix, violation_map)
    print(ovr_k)
    print(cm_k)
    print(sz_k)
    print(n_cls)
    ovr_f, cm_f, sz_f, n_f = compute_cm_fp_strict(violation_map)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    y_range = np.linspace(-0.05, 1.05, 200)
    
    d = _safe_kde(cm_k, y_range)
    if d is not None:
        ax.fill_betweenx(y_range, 1, 1 + d, color='#b5c952', alpha=0.6)
    ax.boxplot([cm_k], positions=[1], widths=0.12, patch_artist=True, showfliers=False,
               medianprops=dict(color='black', linewidth=2), boxprops=dict(color='black'),
               whiskerprops=dict(color='black'), capprops=dict(color='black'))
    jx = np.random.normal(1, 0.04, size=len(cm_k))
    ax.scatter(jx, cm_k, c='#5a6a2a', s=[s*12 for s in sz_k], alpha=0.85, edgecolors='white', linewidth=0.8, zorder=3)
    for x, y, s in zip(jx, cm_k, sz_k):
        ax.annotate(f'n={s}', (x, y), textcoords="offset points", xytext=(10, 0), fontsize=9, color='#444')
    
    ax.boxplot([cm_f], positions=[2], widths=0.12, patch_artist=True, showfliers=False,
               medianprops=dict(color='black', linewidth=2), boxprops=dict(color='black'),
               whiskerprops=dict(color='black'), capprops=dict(color='black'))
    jx = np.random.normal(2, 0.04, size=len(cm_f))
    ax.scatter(jx, cm_f, c='#2a6a5a', s=[s*12 for s in sz_f], alpha=0.85, edgecolors='white', linewidth=0.8, zorder=3)
    for x, y, s in zip(jx, cm_f, sz_f):
        ax.annotate(f'n={s}', (x, y), textcoords="offset points", xytext=(-10, 0), fontsize=9, color='#444', ha='right')
    
    ax.axhline(y=ovr_k, color='crimson', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(1.5, ovr_k + 0.05, f'KM CM = {ovr_k:.4f} | FP CM = {ovr_f:.4f}', ha='center', fontsize=10, color='crimson', fontweight='bold')
    
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(-0.08, 1.15)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Kill-Matrix\n({n_cls} classes)', f'Fingerprint Strict\n({n_f} classes)'], fontsize=11)
    ax.set_ylabel('Class-Level CM', fontsize=11)
    ax.set_title('Softmax: Kill-Matrix vs. Fingerprint Strict Grouping', fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def compute_diagnostic_coverage(kill_matrix, violation_map):
    keys = sorted(kill_matrix.keys())
    # 20 维布尔指纹：0=未触发，非0=触发
    X = np.array([(np.abs(violation_map[k]) > 1e-9).astype(int) for k in keys])
    
    # Kill-Matrix 约简：每个 kill 等价类保留一个代表（取第一个）
    sig_dict = {}
    for idx, k in enumerate(keys):
        sig_dict.setdefault(np.asarray(kill_matrix[k]).tobytes(), []).append(idx)
    km_reps = [g[0] for g in sig_dict.values()]
    km_unique = len(np.unique(X[km_reps], axis=0))
    km_total = len(km_reps)
    
    # Fingerprint Strict 约简：每个指纹类保留一个代表
    fp_dict = {}
    for idx, k in enumerate(keys):
        fp_dict.setdefault(X[idx].tobytes(), []).append(idx)
    fp_reps = [g[0] for g in fp_dict.values()]
    fp_unique = len(np.unique(X[fp_reps], axis=0))
    fp_total = len(fp_reps)
    
    return km_unique, km_total, fp_unique, fp_total

def plot_coverage(kill_matrix, violation_map):
    km_u, km_n, fp_u, fp_n = compute_diagnostic_coverage(kill_matrix, violation_map)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    x = [1, 2]
    heights = [km_u, fp_u]
    colors = ['#b5c952', '#88d8c0']
    bars = ax.bar(x, heights, width=0.5, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, h, n in zip(bars, heights, [km_n, fp_n]):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, 
                f'{h} modes\n({n} classes)', ha='center', fontsize=11, fontweight='bold')
    
    # 画一条“理论上限”线（Fingerprint 的独特模式数 = 上限）
    ax.axhline(y=fp_u, color='teal', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.3, fp_u + 0.2, f'Total distinct modes = {fp_u}', fontsize=10, color='teal')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Kill-Matrix\nReduction', 'Fingerprint\nReduction'], fontsize=12)
    ax.set_ylabel('Unique Diagnostic Fingerprint Patterns Retained', fontsize=12)
    ax.set_title('Diagnostic Coverage after Reduction (Softmax)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(heights) + 2)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



def plot_km_fp_confusion_heatmap(kill_matrix, violation_map, op_name="Softmax", 
                                 save_path=None, figsize=(10, 8), dpi=300):
    """
    绘制 Kill-Matrix 等价类 vs Fingerprint 等价类的混淆矩阵热度图。
    
    说服力：
    - 行 = Kill-Matrix 等价类（size >= 2），标注类规模 n；
    - 列 = Fingerprint 等价类；
    - 若某行横跨多列，直观证明 KM 将多种语义指纹错误归并（Cross-Category Merging）。
    """
    # 1. 构建 Kill-Matrix 等价类
    km_groups = defaultdict(list)
    for name, kills in kill_matrix.items():
        key = tuple(np.asarray(kills).astype(int).tolist())
        km_groups[key].append(name)
    
    # 2. 构建 Fingerprint 等价类
    fp_groups = defaultdict(list)
    for name, fp in violation_map.items():
        key = tuple(np.asarray(fp).round(6).tolist())
        fp_groups[key].append(name)
    
    # 3. 筛选 KM 类：大小 >= 2（这些类才存在"误并"可能）
    km_classes = {k: v for k, v in km_groups.items() if len(v) >= 2}
    fp_classes = {k: v for k, v in fp_groups.items()}
    
    if not km_classes:
        print("No Kill-Matrix equivalence classes with size >= 2.")
        return None
    
    # 4. 构建混淆矩阵
    km_keys = list(km_classes.keys())
    fp_keys = list(fp_classes.keys())
    
    n_km = len(km_keys)
    n_fp = len(fp_keys)
    
    matrix = np.zeros((n_km, n_fp), dtype=int)
    for i, km_key in enumerate(km_keys):
        km_set = set(km_classes[km_key])
        for j, fp_key in enumerate(fp_keys):
            fp_set = set(fp_classes[fp_key])
            matrix[i, j] = len(km_set & fp_set)
    
    # 5. 绘制
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用 YlOrRd，0 显示为接近白色
    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_xticks(np.arange(n_fp))
    ax.set_yticks(np.arange(n_km))
    
    # 标签
    km_labels = [f"KM-{i+1}\n(n={len(km_classes[k])})" for i, k in enumerate(km_keys)]
    fp_labels = [f"FP-{j+1}" for j in range(n_fp)]
    
    ax.set_xticklabels(fp_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(km_labels, fontsize=9)
    
    # 在每个 cell 上标注数字（只标注非零）
    vmax = matrix.max()
    for i in range(n_km):
        for j in range(n_fp):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if val > vmax * 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", 
                       color=text_color, fontsize=7, fontweight='bold')
    
    ax.set_xlabel("Fingerprint Equivalence Classes", fontsize=12)
    ax.set_ylabel("Kill-Matrix Equivalence Classes (size $\\geq$ 2)", fontsize=12)
    ax.set_title(f"Confusion Heatmap: Kill-Matrix vs. Fingerprint ({op_name})\n"
                 f"A row spanning multiple columns = KM merges semantically distinct mutants", 
                 fontsize=13, pad=15)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Mutants', rotation=270, labelpad=18, fontsize=11)
    
    # 网格线
    ax.set_xticks(np.arange(n_fp + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_km + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return matrix, km_classes, fp_classes


def plot_km_layer_heatmap(kill_matrix, violation_map, categories=None, op_name="Softmax",
                          save_path=None, figsize=(8, 6), dpi=300):
    """
    绘制 Kill-Matrix 等价类内部的违规分类触发比例热度图。
    
    参数
    ----
    kill_matrix : dict
        {mutant_name: kill_vector}
    violation_map : dict
        {mutant_name: violation_frequency_vector}，维度为细类数量，不一定固定。
    categories : dict, optional
        高层分类到细类索引的映射，例如：
        {
            'Numerical\nStability': [1, 2, 5, 12, 16, 17, 18],
            'Statistical\nMoments': [8, 9, 10, 11],
            'Distributional\nAxiom': [3, 4, 6, 7, 13, 14],
            'Structural\nInvariants': [0, 15, 19]
        }
        若传入 None，则使用上述默认四层映射。
    """
   
    layer_names = list(categories.keys())
    n_layers = len(categories)
    
    # 构建 KM 等价类（size >= 2）
    km_groups = defaultdict(list)
    for name, kills in kill_matrix.items():
        key = tuple(np.asarray(kills).astype(int).tolist())
        km_groups[key].append(name)
    km_classes = {k: v for k, v in km_groups.items() if len(v) >= 2}
    
    if not km_classes:
        print("No KM classes with size >= 2.")
        return None
    
    km_keys = list(km_classes.keys())
    n_km = len(km_keys)
    
    # 计算每个 KM 类中，各高层分类被触发的变异体比例
    matrix = np.zeros((n_km, n_layers))
    for i, km_key in enumerate(km_keys):
        mutants = km_classes[km_key]
        mutant_layer_triggers = []
        
        for m in mutants:
            v = np.asarray(violation_map[m])
            v_len = len(v)
            layer_vec = np.zeros(n_layers)
            
            # 按 categories 聚合细类 → 高层分类
            for j, indices in enumerate(categories.values()):
                # 动态适配：自动过滤超出当前 violation_map 长度的索引
                valid_indices = [idx for idx in indices if idx < v_len]
                if valid_indices and v[valid_indices].sum() > 0:
                    layer_vec[j] = 1.0      # 该层被触发
            
            mutant_layer_triggers.append(layer_vec)
        
        fps = np.array(mutant_layer_triggers)
        matrix[i, :] = fps.mean(axis=0)   # 该类中触发该层的变异体比例
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(n_layers))
    ax.set_yticks(np.arange(n_km))
    ax.set_xticklabels(layer_names, rotation=0, fontsize=10)
    
    km_labels = [f"KM-{i+1} (n={len(km_classes[k])})" for i, k in enumerate(km_keys)]
    ax.set_yticklabels(km_labels, fontsize=10)
    
    # 格子内标注数值
    for i in range(n_km):
        for j in range(n_layers):
            val = matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                   color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_title(f"Violation-Layer Trigger Rates within Kill-Matrix Classes ({op_name})\n"
                 f"A row with multiple high values = semantic mixture inside one KM class", 
                 fontsize=13, pad=15)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Trigger Proportion', rotation=270, labelpad=18, fontsize=11)
    
    ax.set_xticks(np.arange(n_layers + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_km + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return matrix



# ============================================================
# 统一全局样式配置（三个图共用，放在脚本开头执行一次）
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

# 统一配色
C_PRIMARY = '#2E5C8A'      # 主色（蓝）
C_SECONDARY = '#C0554F'    # 辅色（红）
C_TEXT = '#2B2D42'         # 正文黑
C_GRAY = '#6B7280'         # 辅助灰


# ============================================================
# 图1：McNemar 显著性与 Discordant 对数（双轴图）
# ============================================================
def plot_rq1_McNemar_test_1():
    primitives = ['Softmax', 'LayerNorm', 'RMSNorm', 'MatMul', 'ReLU', 'Sigmoid', 'Attention', 'RBF\nKernel']
    discordant = np.array([42, 33, 51, 8, 6, 15, 37, 151])
    p_vals = np.array([0.0001, 1.16e-10, 4.44e-16, 0.00390625, 0.015625, 3.05e-5, 1.37e-11, 7.5e-46])
    log_p = -np.log10(p_vals)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(primitives))

    # 左轴：-log10(p)
    bars = ax1.bar(x, log_p, width=0.45, color=C_PRIMARY, edgecolor='white', linewidth=0.6, alpha=0.9, zorder=2)
    ax1.set_ylabel(r'$-\log_{10}(p)$', color=C_PRIMARY, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=C_PRIMARY, labelsize=9.5)
    ax1.set_ylim(0, max(log_p) * 1.08)

    # 显著性阈值线（放在底层，避免遮挡柱状图）
    ax1.axhline(y=1.301, color=C_GRAY, linestyle='--', linewidth=0.7, alpha=0.6, zorder=1)
    ax1.axhline(y=2.0, color=C_GRAY, linestyle='-.', linewidth=0.7, alpha=0.6, zorder=1)
    ax1.axhline(y=3.0, color=C_GRAY, linestyle=':', linewidth=0.9, alpha=0.7, zorder=1)
    # 阈值标签放在图右侧，不遮挡数据
    ax1.text(7.35, 1.6, r'$\alpha=0.05$', fontsize=8, color=C_GRAY, ha='left')
    ax1.text(7.35, 2.3, r'$\alpha=0.01$', fontsize=8, color=C_GRAY, ha='left')
    ax1.text(7.35, 3.3, r'$\alpha=0.001$', fontsize=8, color=C_GRAY, ha='left')

    # 右轴：discordant pairs
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.plot(x, discordant, 'D-', color=C_SECONDARY, linewidth=1.8, markersize=6,
             markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=C_SECONDARY, zorder=3)
    ax2.set_ylabel('Discordant Pairs', color=C_SECONDARY, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=C_SECONDARY, labelsize=9.5)
    ax2.set_ylim(0, max(discordant) * 1.12)

    # 统一标注风格
    for i, (lp, d) in enumerate(zip(log_p, discordant)):
        ax1.text(i, lp + 1.0, f'{lp:.1f}', ha='center', va='bottom', fontsize=8.5,
                 color=C_PRIMARY, fontweight='bold')
        ax2.text(i, d + 3, str(d), ha='center', va='bottom', fontsize=8.5, color=C_SECONDARY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(primitives, fontsize=9.5)
    ax1.set_title('McNemar Test Significance Across Primitives', fontweight='bold',
                  fontsize=13, color=C_TEXT, pad=12)

    # 图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_PRIMARY, alpha=0.9, label=r'$-\log_{10}(p)$'),
        Line2D([0], [0], color=C_SECONDARY, marker='D', linestyle='-', markerfacecolor='white',
               markeredgewidth=1.8, markersize=6, label='Discordant Pairs')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig('rq1_mcnemar_test_1.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# ============================================================
# 图2：div_only vs kill_only（水平棒棒糖图）
# ============================================================
def plot_rq1_McNemar_test_2():
    primitives = ['Softmax', 'LayerNorm', 'RMSNorm', 'MatMul', 'ReLU', 'Sigmoid', 'Attention', 'RBF Kernel']
    div_only = np.array([35, 33, 51, 8, 6, 15, 37, 151])
    kill_only = np.array([7, 0, 0, 0, 0, 0, 0, 0])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(primitives))

    # div_only: 水平棒棒糖
    for i, d in enumerate(div_only):
        ax.plot([0, d], [i, i], color=C_PRIMARY, linewidth=1.5, alpha=0.5, zorder=1)
        ax.scatter(d, i, color=C_PRIMARY, s=70, zorder=3, edgecolor='white', linewidth=1.2)

    # kill_only: 标记符号
    for i, k in enumerate(kill_only):
        if k == 0:
            ax.scatter(0, i, marker='x', color=C_GRAY, s=55, zorder=4, linewidth=2)
        else:
            ax.scatter(k, i, marker='X', color=C_SECONDARY, s=85, zorder=4,
                       edgecolor='white', linewidth=1.2)

    # 统一标注风格
    for i, (d, k) in enumerate(zip(div_only, kill_only)):
        ax.text(d + 3, i, str(d), va='center', fontsize=8.5, color=C_PRIMARY, fontweight='bold')
        if k > 0:
            ax.text(k + 1.5, i - 0.22, str(k), va='center', fontsize=8, color=C_SECONDARY, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(primitives, fontsize=9.5)
    ax.set_xlabel('Number of Discordant Pairs', fontsize=11)
    ax.set_title('Decomposition of Discordant Pairs', fontweight='bold',
                 fontsize=13, color=C_TEXT, pad=12)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_PRIMARY,
               markersize=9, label='Diff only by Fingerprint (div_only)',
               markeredgecolor='white'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=C_SECONDARY,
               markersize=9, label='Diff only by Kill-Matrix (kill_only)',
               markeredgecolor='white'),
        Line2D([0], [0], marker='x', color=C_GRAY, markersize=7, linestyle='None',
               label='kill_only = 0')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=8.5)

    ax.set_xlim(-2, max(div_only) * 1.12)
    ax.set_ylim(-0.5, len(primitives) - 0.5)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.25)

    plt.tight_layout()
    plt.savefig('rq1_mcnemar_test_2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# ============================================================
# 图3：Softmax 2×2 列联表热力图
# ============================================================
def plot_rq1_McNemar_test_3():
    contingency = np.array([[482, 7], [35, 96]])
    labels = np.array([['both_diff\n482', 'kill_only\n7'],
                       ['div_only\n35', 'both_same\n96']])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(contingency, annot=labels, fmt='', cmap='Blues',
                cbar_kws={'label': 'Pair Count'},
                xticklabels=['Fingerprint: Diff', 'Fingerprint: Same'],
                yticklabels=['Kill-Matrix: Diff', 'Kill-Matrix: Same'],
                ax=ax, linewidths=2.5, linecolor='white', annot_kws={'size': 11})

    ax.set_title('Softmax: Judgment Contingency Table\n(620 Independent Pairs)',
                 fontweight='bold', fontsize=13, color=C_TEXT, pad=12)
    plt.xticks(rotation=0, ha='center', fontsize=9.5)
    plt.yticks(rotation=0, va='center', fontsize=9.5)

    # 边际标注
    ax.text(2.12, 0.5, '489', fontsize=10, va='center', color=C_TEXT, fontweight='bold')
    ax.text(2.12, 1.5, '131', fontsize=10, va='center', color=C_TEXT, fontweight='bold')
    ax.text(0.5, 2.12, '517', fontsize=10, ha='center', color=C_TEXT, fontweight='bold')
    ax.text(1.5, 2.12, '103', fontsize=10, ha='center', color=C_TEXT, fontweight='bold')
    ax.text(2.12, 2.12, '620', fontsize=10, ha='center', va='center',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_PRIMARY, edgecolor='none'))

    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Pair Count', fontsize=10)

    plt.tight_layout()
    plt.savefig('rq1_mcnemar_test_3.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_rq1_McNemar_test():
    plot_rq1_McNemar_test_1()
    plot_rq1_McNemar_test_2()
    plot_rq1_McNemar_test_3()
