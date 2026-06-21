import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

def compute_rq2_metrics(kill_matrix, violation_map, categories, n_fine=20):
    """
    方案 A：分辨率与压缩指标（基于四层聚合指纹，不含 NMI）
    
    参数
    ----
    kill_matrix : dict[str, list[int]]
        变异体名 -> 二元 kill 向量 (0/1)
    violation_map : dict[str, list[float] | np.ndarray]
        变异体名 -> (20,) 细类违规频次向量
    categories : dict[str, list[int]]
        四层分类映射，如 {
            'Numerical Stability': [1,2,5,12,16,17,18],
            'Statistical Moments': [8,9,10,11],
            'Distributional Axiom': [3,4,6,7,13,14],
            'Structural Invariants': [0,15,19]
        }
    n_fine : int
        细类维度，默认 20
    
    返回
    ----
    dict : M, Uk, Uf, RG, KCR, FCR, CPR
    """
    names = list(kill_matrix.keys())
    M = len(names)
    n_fine=np.array(list(violation_map.values())).shape[1]
    # 1. Kill-Matrix 唯一模式数（二元 kill 向量，不变）
    km_patterns = set()
    for n in names:
        km_key = tuple(np.asarray(kill_matrix[n]).astype(int).tolist())
        km_patterns.add(km_key)
    Uk = len(km_patterns)
    
    # 2. 将细类 (20,) 聚合为四层 (4,) 指纹，再提取唯一模式
    fp_patterns = set()
    for n in names:
        fine_vec = np.asarray(violation_map[n], dtype=float)
        if fine_vec.shape != (n_fine,):
            raise ValueError(f"变异体 {n}: 期望 ({n_fine},) 向量，得到 {fine_vec.shape}")
        
        # 聚合到四层
        layer_counts = np.zeros(4)
        for i, indices in enumerate(categories.values()):
            valid_idx = [idx for idx in indices if 0 <= idx < n_fine]
            layer_counts[i] = fine_vec[valid_idx].sum()
        
        # 以聚合后的四维向量作为指纹，精度 1e-6
        fp_key = tuple(layer_counts.round(6).tolist())
        fp_patterns.add(fp_key)
    
    Uf = len(fp_patterns)
    
    # 3. 核心指标
    RG  = Uf / Uk if Uk > 0 else 0.0   # Resolution Gain（分辨率增益）
    KCR = 1.0 - Uk / M                  # Kill-Matrix 碰撞率
    FCR = 1.0 - Uf / M                  # Fingerprint 碰撞率（Strict，配合 CM=0）
    CPR = Uf / M                        # 压缩率（越低约简越激进）
    
    return {
        'M': M,                # 总变异体数
        'Uk': Uk,              # Kill-Matrix 唯一模式数
        'Uf': Uf,              # Fingerprint 唯一模式数（四层聚合后）
        'RG': round(RG, 3),    # 分辨率增益（指纹 / KM）
        'KCR': round(KCR, 3),  # Kill-Matrix 碰撞率（越低越好）
        'FCR': round(FCR, 3),  # Fingerprint 碰撞率（配合 CM=0 使用）
        'CPR': round(CPR, 3),  # 压缩率（越低约简越激进）
    }

def aggregate_to_4layers(violation_entry, categories, n_fine=20):
    """将 (20,) 细类频次向量聚合为 (4,) 四层向量"""
    arr = np.asarray(violation_entry, dtype=float)
    if arr.shape != (n_fine,):
        raise ValueError(f"期望 (20,) 向量，得到 {arr.shape}")
    
    layer_counts = np.zeros(4)
    layer_names = list(categories.keys())
    for i, indices in enumerate(categories.values()):
        valid_idx = [idx for idx in indices if 0 <= idx < n_fine]
        layer_counts[i] = arr[valid_idx].sum()
    return layer_counts, layer_names

def plot_fingerprint_tsne(violation_map, categories, op_name="Softmax", 
                                     save_path=None, figsize=(11, 9), dpi=300):
    """
    多层叠加圆 t-SNE：每层独立用圆大小表示触发频次，避免 argmax 硬标签淹没次要层。
    """
    # ===== 新增：自动创建保存目录兜底 =====
    import os
    # ========== 修复目录创建代码 ==========
    if save_path is not None:
        folder = os.path.dirname(save_path)
        # 无论是否存在，直接创建，exist_ok=True不存在自动建
        os.makedirs(folder, exist_ok=True)
    # ======================================

    names = list(violation_map.keys())
    M = len(names)
    
    # 聚合为 (M, 4)
    fp_list = []
    n_fine=np.array(list(violation_map.values())).shape[1]
    for n in names:
        vec, layer_names = aggregate_to_4layers(violation_map[n], categories,n_fine=n_fine)
        fp_list.append(vec)
    X_raw = np.array(fp_list)  # (M, 4)
    
    print(f"[{op_name}] 聚合后指纹矩阵: {X_raw.shape}")
    print(f"  四层总触发: {X_raw.sum(axis=0)}")
    print(f"  每层至少触发一次的变异体数: {(X_raw > 0).sum(axis=0)}")
    
    # 归一化频率向量（用于 t-SNE）
    row_sums = X_raw.sum(axis=1, keepdims=True)
    X_norm = X_raw / (row_sums + 1e-9)
    
    # 全零保护
    zero_mask = (row_sums.squeeze() == 0)
    n_zero = zero_mask.sum()
    if n_zero > 0:
        print(f"  警告: {n_zero} 个变异体全零，赋予微小均匀扰动")
        X_norm[zero_mask] = 0.25
    
    # t-SNE
    perplexity = min(30, M - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                init='pca', learning_rate='auto')
    emb = tsne.fit_transform(X_norm)
    
    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # 红蓝绿紫
    
    # 按总触发频次排序，先画高频层（大圆），再画低频层（小圆），避免小圆被完全覆盖
    total_by_layer = X_raw.sum(axis=0)
    layer_order = np.argsort(-total_by_layer)
    
    for idx in layer_order:
        freqs = X_raw[:, idx]
        mask = freqs > 0
        if mask.sum() == 0:
            continue
        
        # 圆大小：sqrt(freq) * scale，压缩动态范围，确保低频层可见
        sizes = np.sqrt(freqs[mask]) * 60  
        
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=colors[idx], s=sizes, alpha=0.45,
                   label=f"{layer_names[idx]}  (n_triggered={mask.sum()}/{M})",
                   edgecolors='none', zorder=idx)
    
    # 黑色小点：所有变异体的位置锚点，防止全零变异体消失
    ax.scatter(emb[:, 0], emb[:, 1], c='black', s=8, alpha=0.4, zorder=10, label='Mutant position')
    
    # 标注全零变异体
    if n_zero > 0:
        ax.scatter(emb[zero_mask, 0], emb[zero_mask, 1], 
                   c='black', s=120, marker='x', linewidths=2, zorder=11, label=f'No violation (n={n_zero})')
    
    ax.set_title(f"Fingerprint Space t-SNE ({op_name})\n"
                 f"Multi-layer overlay: circle size ∝ trigger frequency, color = violation layer", 
                 fontsize=13)
    ax.legend(title="Violation Layer", loc='best', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return emb, X_raw

def extract_case_studies(kill_matrix, violation_map, categories, n_fine=20, max_cases=3):
    """
    基于四层聚合指纹，抽取 Kill-Matrix 同组但指纹不同的典型案例。
    """
    names = list(kill_matrix.keys())
    layer_names = list(categories.keys())  # ['Numerical Stability', 'Statistical Moments', ...]
    n_fine=np.array(list(violation_map.values())).shape[1]
    # 预计算每个变异体的四层聚合指纹
    fp_4layer = {}
    for n in names:
        fine_vec = np.asarray(violation_map[n], dtype=float)
        if fine_vec.shape != (n_fine,):
            raise ValueError(f"变异体 {n}: 期望 ({n_fine},) 向量，得到 {fine_vec.shape}")
        
        layer_counts = np.zeros(4)
        for i, indices in enumerate(categories.values()):
            valid_idx = [idx for idx in indices if 0 <= idx < n_fine]
            layer_counts[i] = fine_vec[valid_idx].sum()
        
        fp_4layer[n] = tuple(layer_counts.round(6).tolist())
    
    # 构建 Kill-Matrix 等价类
    km_groups = defaultdict(list)
    for n in names:
        km_key = tuple(np.array(kill_matrix[n]).astype(int).tolist())
        km_groups[km_key].append(n)
    
    cases = []
    
    # 找 KM 同组但四层指纹不同的配对
    for km_key, members in km_groups.items():
        if len(members) < 2:
            continue
        
        unique_fps = set(fp_4layer[m] for m in members)
        if len(unique_fps) > 1:
            # 找四层指纹 L1 距离最大的两个变异体
            best_pair = None
            best_dist = -1
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    m1, m2 = members[i], members[j]
                    d = sum(abs(np.array(fp_4layer[m1]) - np.array(fp_4layer[m2])))
                    if d > best_dist:
                        best_dist = d
                        best_pair = (m1, m2)
            
            if best_pair:
                m1, m2 = best_pair
                fp1 = np.array(fp_4layer[m1])
                fp2 = np.array(fp_4layer[m2])
                
                dom_idx_1 = int(np.argmax(fp1))
                dom_idx_2 = int(np.argmax(fp2))
                
                cases.append({
                    'type': 'KM_merged_FP_split',
                    'm1': m1,
                    'm2': m2,
                    'km_pattern': km_key,
                    'fp_m1': fp_4layer[m1],
                    'fp_m2': fp_4layer[m2],
                    'dominant_m1': dom_idx_1,               # int: 0,1,2,3
                    'dominant_m2': dom_idx_2,               # int: 0,1,2,3
                    'layer_name_m1': layer_names[dom_idx_1], # str: e.g. "Numerical Stability"
                    'layer_name_m2': layer_names[dom_idx_2], # str: e.g. "Distributional Axiom"
                    'l1_distance': round(float(best_dist), 3)
                })
            
            if len(cases) >= max_cases:
                break
    
    return cases

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cases(save_dir="rq2", dpi=600):

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    cases = {
        "Case 1\nRecipe Difference": {
            "M32": [231, 162, 372, 30],
            "M49": [256, 162, 343, 30],
            "km": "(0,0,0,0,0...)"
        },

        "Case 2\nFalse Relative": {
            "M14": [173, 92, 157, 23],
            "M37": [188, 95, 371, 24],
            "km": "(0,1,1,1,1...)"
        },

        "Case 3\nTrue Equivalence": {
            "M17": [255, 173, 400, 29],
            "M45": [254, 173, 398, 29],
            "km": "(0,1,1,1,1...)"
        }
    }

    labels = [
        "L1\nNumerical",
        "L2\nBoundary",
        "L3\nDistribution",
        "L4\nCross-Input"
    ]

    N = len(labels)

    angles = np.linspace(
        0,
        2*np.pi,
        N,
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig, axes = plt.subplots(
        1,
        3,
        subplot_kw=dict(polar=True),
        figsize=(15, 5)
    )

    colors = [
        "#4C72B0",
        "#DD8452"
    ]

    for ax, (title, data) in zip(axes, cases.items()):

        mutant_names = list(data.keys())[:2]

        fp1 = data[mutant_names[0]]
        fp2 = data[mutant_names[1]]

        values1 = fp1 + fp1[:1]
        values2 = fp2 + fp2[:1]

        ax.plot(
            angles,
            values1,
            linewidth=2.5,
            color=colors[0],
            label=mutant_names[0]
        )

        ax.fill(
            angles,
            values1,
            alpha=0.20,
            color=colors[0]
        )

        ax.plot(
            angles,
            values2,
            linewidth=2.5,
            color=colors[1],
            label=mutant_names[1]
        )

        ax.fill(
            angles,
            values2,
            alpha=0.20,
            color=colors[1]
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        ax.set_title(
            title,
            y=1.18,
            fontweight="bold"
        )

        ax.grid(alpha=0.4)

        dist = np.linalg.norm(
            np.array(fp1) - np.array(fp2)
        )

        ax.text(
            0.5,
            -0.25,
            f"KM Class: {data['km']}\nL2 Distance = {dist:.1f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9
        )

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.25, 1.15),
            frameon=False
        )

    fig.suptitle(
        "Fingerprint Analysis Inside Kill-Matrix Equivalence Classes",
        fontsize=14,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    pdf_file = os.path.join(
        save_dir,
        "rq2_case_radar.pdf"
    )

    png_file = os.path.join(
        save_dir,
        "rq2_case_radar.png"
    )

    plt.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.savefig(
        png_file,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:")
    print(pdf_file)
    print(png_file)


















    