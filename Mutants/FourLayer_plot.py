import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis('off')

# 颜色定义
c_l1 = '#BBDEFB'; c_l2 = '#C8E6C9'; c_l3 = '#FFF9C4'; c_l4 = '#FFCDD2'
c_text = '#333333'; c_math = '#1565C0'; c_warn = '#E65100'

# ========== 标题 ==========
ax.text(6.5, 7.6, 'Mathematical Justification of the Four-Layer Taxonomy', 
        fontsize=16, fontweight='bold', ha='center', color=c_text)

# ========== 中部：四层作为数学对象的偏序结构 ==========
# 用数学对象标注每层：Scalar Field -> Moment Functional -> Probability Measure -> Group Orbit
layer_math_obj = [
    ('L1', 'Numerical\nStability', c_l1, 'Scalar Field\n$\mathbb{R}^{d}$', '5 cats'),
    ('L2', 'Statistical\nMoments', c_l2, 'Moment Functional\n$\mu, \sigma^2, RMS$', '4 cats'),
    ('L3', 'Distributional\nAxioms', c_l3, 'Probability Measure\n$\mathcal{P}(\Omega)$', '3 cats'),
    ('L4', 'Structural\nInvariants', c_l4, 'Group Orbit\n$G \cdot x$', '6 cats')
]

x_center = 6.5
y_positions = [1.2, 2.6, 4.0, 5.4]
box_w = 2.8; box_h = 1.0

for i, (label, name, color, math_obj, count) in enumerate(layer_math_obj):
    y = y_positions[i]
    # 主框
    rect = FancyBboxPatch((x_center - box_w/2, y), box_w, box_h, 
                          boxstyle="round,pad=0.03", facecolor=color, 
                          edgecolor='#555555', linewidth=1.5)
    ax.add_patch(rect)
    
    # 层名
    ax.text(x_center - box_w/2 + 0.15, y + box_h - 0.15, 
            f'{label} {name}', fontsize=11, fontweight='bold', 
            va='top', ha='left', color=c_math)
    
    # 数学对象（核心：说明每层检验的数学结构不同）
    ax.text(x_center, y + 0.35, math_obj, fontsize=9, 
            va='center', ha='center', color=c_math, style='italic')
    
    # 类别数
    ax.text(x_center + box_w/2 - 0.15, y + box_h - 0.15, count, 
            fontsize=10, va='top', ha='right', color=c_text, fontweight='bold')

# 绘制逻辑蕴含箭头（偏序关系）
for i in range(3):
    y_bottom = y_positions[i] + box_h
    y_top = y_positions[i+1]
    ax.annotate('', xy=(x_center, y_top), xytext=(x_center, y_bottom),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2))
    # 蕴含标签
    labels = ['$\mathcal{M} \subseteq \mathbb{R}^d$\nrequires valid scalars',
              '$\mathcal{P} \in \mathcal{M}_1^+$\nrequires correct moments',
              '$G \cdot x$\nrequires valid $\mathcal{P}$']
    ax.text(x_center + 0.3, (y_bottom + y_top)/2, labels[i], 
            fontsize=8, va='center', ha='left', color='#666666')

# ========== 左侧：极小完备性反证（为什么不是3层或5层） ==========
# 3层方案的问题
ax.text(1.5, 6.8, 'Why not 3 layers?', fontsize=12, fontweight='bold', ha='center', color=c_warn)
rect_3 = FancyBboxPatch((0.3, 4.8), 2.4, 1.6, boxstyle="round,pad=0.03",
                        facecolor='#FFF3E0', edgecolor=c_warn, linewidth=1.5, linestyle='--')
ax.add_patch(rect_3)
ax.text(1.5, 6.1, 'Missing L4:\nStructural Invariants', fontsize=10, fontweight='bold', 
        ha='center', va='top', color=c_warn)
ax.text(1.5, 5.5, 'Counter-example:\nSoftmax passes L3\n(prob sum=1),\nbut breaks under\n$x \\to x+c$\n(max-trick bug)', 
        fontsize=8.5, ha='center', va='top', color=c_text, linespacing=1.3)

# 5层方案的问题
ax.text(1.5, 3.8, 'Why not 5 layers?', fontsize=12, fontweight='bold', ha='center', color=c_warn)
rect_5 = FancyBboxPatch((0.3, 1.8), 2.4, 1.6, boxstyle="round,pad=0.03",
                        facecolor='#FFF3E0', edgecolor=c_warn, linewidth=1.5, linestyle='--')
ax.add_patch(rect_5)
ax.text(1.5, 3.1, 'Extra layer would be:\nImplementation-level', fontsize=10, fontweight='bold', 
        ha='center', va='top', color=c_warn)
ax.text(1.5, 2.5, 'e.g., Keepdim,\nBroadcasting, Shape\n→ Not mathematical\ninvariants; already\ncovered by unit tests', 
        fontsize=8.5, ha='center', va='top', color=c_text, linespacing=1.3)

# ========== 右侧：完备性覆盖（9算子的失效模式如何被4层完全覆盖） ==========
ax.text(11.5, 6.8, 'Completeness Coverage', fontsize=12, fontweight='bold', ha='center', color=c_math)

# 画一个 4x9 的覆盖矩阵示意（简化版，只展示层与算子家族的映射）
families = ['Softmax', 'RMSNorm', 'LayerNorm', 'ReLU', 'Sigmoid', 'GELU', 'MatMul', 'RBF', 'Attention']
family_colors = ['#E3F2FD', '#E8F5E9', '#E8F5E9', '#F3E5F5', '#F3E5F5', '#F3E5F5', '#FFF3E0', '#FFF3E0', '#E3F2FD']
y_fam_start = 5.8
for idx, (fam, fc) in enumerate(zip(families, family_colors)):
    y = y_fam_start - idx * 0.55
    # 算子名
    rect = Rectangle((9.8, y), 1.6, 0.4, facecolor=fc, edgecolor='#333333', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(10.6, y+0.2, fam, fontsize=8.5, ha='center', va='center', fontweight='bold')
    
    # 覆盖的层（用彩色圆点表示）
    coverage = {
        'Softmax': [c_l1, c_l3, c_l4],
        'RMSNorm': [c_l1, c_l2, c_l4],
        'LayerNorm': [c_l1, c_l2, c_l4],
        'ReLU': [c_l1, c_l4],
        'Sigmoid': [c_l1, c_l3, c_l4],
        'GELU': [c_l1, c_l3, c_l4],
        'MatMul': [c_l1, c_l4],
        'RBF': [c_l1, c_l4],
        'Attention': [c_l1, c_l3, c_l4]
    }
    x_dot = 11.8
    for c in coverage[fam]:
        circle = Circle((x_dot, y+0.2), 0.12, facecolor=c, edgecolor='#333333', linewidth=0.5)
        ax.add_patch(circle)
        x_dot += 0.35

ax.text(11.5, 6.3, "Each operator\\'s failure modes\\nare fully covered by\\na subset of layers", 
        fontsize=9, ha='center', va='top', color=c_text, style='italic')

# 图例
legend_x = 9.8; legend_y = 1.3
ax.text(legend_x, legend_y + 0.6, 'Layer coverage:', fontsize=9, fontweight='bold', color=c_text)
for i, (label, color) in enumerate(zip(['L1', 'L2', 'L3', 'L4'], [c_l1, c_l2, c_l3, c_l4])):
    circle = Circle((legend_x + i*0.6, legend_y + 0.2), 0.12, facecolor=color, edgecolor='#333333')
    ax.add_patch(circle)
    ax.text(legend_x + i*0.6, legend_y - 0.15, label, fontsize=8, ha='center', color=c_text)

# ========== 底部中心：核心结论 ==========
ax.text(6.5, 0.4, 
        "Conclusion: 4 layers form a mathematically minimal complete chain (no redundancy, no omission); '\n        '18 categories exhaustively partition the mathematical invariant violation space for all 9 operator families.",
        fontsize=10, ha='center', va='center', color=c_text, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor=c_math, linewidth=1.5))

plt.tight_layout()
# plt.savefig('/mnt/agents/output/taxonomy_mathematical_justification.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()