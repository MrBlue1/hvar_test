import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 输入示例数据
# -----------------------------
# 支持多个算子和测试规模
# 数据结构：
# {算子: {测试规模: {机制: 四格表}}}
data = {
    'softmax': {
        50: {
            'original': {'both_diff': 400, 'kill_only': 10, 'div_only': 25, 'both_same': 70},
            'triple':   {'both_diff': 400, 'kill_only': 35, 'div_only': 0, 'both_same': 70}
        },
        100: {
            'original': {'both_diff': 800, 'kill_only': 20, 'div_only': 50, 'both_same': 140},
            'triple':   {'both_diff': 800, 'kill_only': 70, 'div_only': 0, 'both_same': 140}
        },
        200: {
            'original': {'both_diff': 1540, 'kill_only': 24, 'div_only': 96, 'both_same': 293},
            'triple':   {'both_diff': 1540, 'kill_only': 120, 'div_only': 0, 'both_same': 293}
        }
    },
    'layerNorm': {
        50: {
            'original': {'both_diff': 350, 'kill_only': 15, 'div_only': 30, 'both_same': 55},
            'triple':   {'both_diff': 350, 'kill_only': 45, 'div_only': 0, 'both_same': 55}
        }
        # 可以继续添加更多规模
    }
}

# -----------------------------
# 1️⃣ 原 vs 三层决策柱状对比图
# -----------------------------
def plot_fourfold_comparison(operator, scale):
    counts = data[operator][scale]
    labels = ['both_diff', 'kill_only', 'div_only', 'both_same']
    orig_vals = [counts['original'][l] for l in labels]
    triple_vals = [counts['triple'][l] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x - width/2, orig_vals, width, label='Original Kill', color='skyblue')
    rects2 = ax.bar(x + width/2, triple_vals, width, label='Triple Decision', color='salmon')
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('配对数量')
    ax.set_title(f'{operator} 四格表对比（测试规模={scale}）')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 示例调用
plot_fourfold_comparison('softmax', 200)

# -----------------------------
# 2️⃣ div_only / kill_only 随测试规模变化折线图
# -----------------------------
def plot_div_kill_vs_scale(operator):
    scales = sorted(data[operator].keys())
    div_only_orig = [data[operator][s]['original']['div_only'] for s in scales]
    kill_only_orig = [data[operator][s]['original']['kill_only'] for s in scales]
    div_only_triple = [data[operator][s]['triple']['div_only'] for s in scales]
    kill_only_triple = [data[operator][s]['triple']['kill_only'] for s in scales]
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(scales, div_only_orig, 'o-', label='div_only (Original)', color='blue')
    ax.plot(scales, kill_only_orig, 's--', label='kill_only (Original)', color='cyan')
    ax.plot(scales, div_only_triple, 'o-', label='div_only (Triple)', color='red')
    ax.plot(scales, kill_only_triple, 's--', label='kill_only (Triple)', color='orange')
    
    ax.set_xlabel('测试规模')
    ax.set_ylabel('配对数量')
    ax.set_title(f'{operator} div_only / kill_only 随测试规模变化')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 示例调用
plot_div_kill_vs_scale('softmax')

# -----------------------------
# 3️⃣ 百分比堆叠图：四格表占比
# -----------------------------
def plot_fourfold_stacked(operator, scale):
    counts = data[operator][scale]
    labels = ['both_diff', 'kill_only', 'div_only', 'both_same']
    
    orig_vals = np.array([counts['original'][l] for l in labels])
    triple_vals = np.array([counts['triple'][l] for l in labels])
    
    # 计算占比
    orig_frac = orig_vals / orig_vals.sum()
    triple_frac = triple_vals / triple_vals.sum()
    
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6,5))
    
    bottom_orig = np.zeros(2)
    bottom_triple = np.zeros(2)
    colors = ['skyblue', 'lightgreen', 'gold', 'lightgray']
    
    for i, label in enumerate(labels):
        ax.bar(x[0], orig_frac[i], bottom=bottom_orig[0], color=colors[i], label=label if x[0]==0 else "")
        ax.bar(x[1], triple_frac[i], bottom=bottom_triple[1], color=colors[i])
        bottom_orig[0] += orig_frac[i]
        bottom_triple[1] += triple_frac[i]
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Original Kill', 'Triple Decision'])
    ax.set_ylabel('占比')
    ax.set_title(f'{operator} 四格表占比（测试规模={scale}）')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 示例调用
plot_fourfold_stacked('softmax', 200)