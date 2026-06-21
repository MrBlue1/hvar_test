
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')

colors = {
    'start_end': '#1f4e79',
    'process': '#2e75b6',
    'subprocess': '#5b9bd5',
    'decision': '#c55a11',
    'data': '#70ad47',
    'arrow': '#404040',
    'text': '#ffffff',
    'subtext': '#000000'
}

def draw_box(ax, x, y, width, height, text, color, text_color='white', fontsize=10, bold=True):
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, weight=weight, wrap=True)
    return box

def draw_diamond(ax, x, y, size, text, color):
    diamond = plt.Polygon([(x, y+size), (x+size*1.2, y), (x, y-size), (x-size*1.2, y)], 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, 
            color='white', weight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=None):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color=colors['arrow'])
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y, label, fontsize=9, color='red', weight='bold')

# 标题
ax.text(7, 17.2, '基于行为语义的AI算子变异体约简方法流程图', 
        fontsize=16, weight='bold', ha='center', color=colors['start_end'])

# S1: 测试用例生成
draw_box(ax, 7, 16, 5, 0.8, 'S1: 生成变异体测试用例集合T\n(LHS拉丁超立方采样)', colors['process'])

# S2: 程序加载
draw_box(ax, 7, 14.8, 5, 0.8, 'S2: 加载源程序F0和变异体程序集合F', colors['process'])

# S3: 双向量生成 - 使用ASCII表示法避免Unicode下标
draw_box(ax, 7, 13.2, 5.5, 1.2, 'S3: 执行测试用例，生成双向量\nMS向量V^MS (dim=2n)  行为语义向量V^B (dim=dn)', colors['process'])

# S3子步骤细节 - 使用普通文本
ax.text(2.5, 13.2, 'S3.1: 输出一致性判断\n- 定义阈值e=1e-6\n- 判断存活状态s_ki', 
        fontsize=9, ha='center', va='center', 
        bbox=dict(boxstyle='round', facecolor=colors['subprocess'], alpha=0.8))

ax.text(11.5, 13.2, 'S3.2: 行为语义分类\n- AST解析提取标签\n- d类行为语义L={l0...ld}', 
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor=colors['subprocess'], alpha=0.8))

# S4: 矩阵构建 - 使用×代替×，使用^代替上标
draw_box(ax, 7, 11.5, 5.5, 0.9, 'S4: 构建矩阵\n杀死矩阵 K (m x n)    行为语义矩阵 P (m x d)', colors['process'])

# S5: 前置诊断
draw_diamond(ax, 7, 9.8, 0.7, 'S5: 前置诊断\n评估数据质量', colors['decision'])

# S5 指标说明
ax.text(2.5, 9.8, 'S5.1: 测试集完备性\n(Jaccard相似度)', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='#f4b084', alpha=0.8))
ax.text(11.5, 9.8, 'S5.2: 模态相关性\n(互信息MI)\nS5.3: 行为多样性\n(分布熵H)', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='#f4b084', alpha=0.8))

# S6: 聚类约简
draw_box(ax, 7, 7.8, 5.5, 1.3, 'S6: 行为约束聚类约简\n行为预分组 -> 组内分层选优 -> 代表集合F^R', colors['data'])

# S6 子步骤
ax.text(2.3, 7.8, 'S6.1: 行为预分组\nG={G0,G1,...Gd}', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='#a9d18e', alpha=0.8))
ax.text(11.7, 7.8, 'S6.2: 组内分层选优\n|Gj|<=3: 选最大杀死数\n|Gj|>3: K-Means聚类', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='#a9d18e', alpha=0.8))

# S7: 效果评估
draw_box(ax, 7, 5.8, 5, 0.9, 'S7: 评估约简效果\nFTRR行为保留率 | MS杀死率 | 约简比R', colors['process'])

# 输出结果
draw_box(ax, 7, 4.2, 4, 0.8, '输出: 变异体代表集合F^R\n约简效果评估报告', colors['start_end'])

# 绘制连接箭头
draw_arrow(ax, 7, 15.6, 7, 14.4)
draw_arrow(ax, 7, 14.4, 7, 13.8)
draw_arrow(ax, 7, 12.6, 7, 11.95)

# S3到子步骤的连接
ax.plot([4.5, 5.5], [13.2, 13.2], 'k--', alpha=0.5)
ax.plot([8.5, 9.5], [13.2, 13.2], 'k--', alpha=0.5)

draw_arrow(ax, 7, 11.05, 7, 10.5)

# S5决策分支
ax.annotate('', xy=(4.5, 9.8), xytext=(6.3, 9.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(5.4, 10.0, '计算指标', fontsize=8, color='gray')
ax.annotate('', xy=(9.5, 9.8), xytext=(7.7, 9.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

draw_arrow(ax, 7, 9.1, 7, 8.45)

# S6到子步骤的连接
ax.plot([4.0, 5.0], [7.8, 7.8], 'k--', alpha=0.5)
ax.plot([9.0, 10.0], [7.8, 7.8], 'k--', alpha=0.5)

draw_arrow(ax, 7, 7.15, 7, 6.25)
draw_arrow(ax, 7, 5.35, 7, 4.6)

# 添加左侧阶段标注
ax.text(0.8, 16, '输入阶段', fontsize=11, weight='bold', rotation=90, va='center', color=colors['process'])
ax.text(0.8, 12.5, '特征提取', fontsize=11, weight='bold', rotation=90, va='center', color=colors['process'])
ax.text(0.8, 9.8, '质量诊断', fontsize=11, weight='bold', rotation=90, va='center', color=colors['decision'])
ax.text(0.8, 6.5, '约简核心', fontsize=11, weight='bold', rotation=90, va='center', color=colors['data'])
ax.text(0.8, 4.5, '输出阶段', fontsize=11, weight='bold', rotation=90, va='center', color=colors['start_end'])

# 图例
legend_y = 2.5
ax.text(7, legend_y+0.5, '图例说明:', fontsize=11, weight='bold', ha='center')
legend_items = [
    (colors['process'], '主要步骤 (S1-S4, S7)'),
    (colors['decision'], '诊断决策 (S5)'),
    (colors['data'], '核心算法 (S6)'),
    (colors['subprocess'], '子步骤细节')
]

for i, (color, label) in enumerate(legend_items):
    x_pos = 3 + i*3
    rect = plt.Rectangle((x_pos-0.3, legend_y-0.2), 0.6, 0.4, 
                         facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x_pos+0.5, legend_y, label, fontsize=9, va='center')

plt.tight_layout()
plt.savefig('../docs/patent_flowchart_fixed.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
print("已修复字体警告，新版流程图生成完成")
