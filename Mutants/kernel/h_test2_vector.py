"""
Mutant Behavior Fingerprint + Clustering Reduction
Based on previous Layered Decision Engine
"""

import numpy as np
from sklearn.cluster import KMeans
import importlib
import os

from h_test2 import check_constraints, generate_lhs_tests, TEST_CASE_NUM, DIM, POINTS

# -----------------------------
# 1. 构建行为指纹
# -----------------------------
def build_fingerprint(mut_func, ref_func, tests):
    vectors = []

    for X, gamma in tests:
        try:
            K_mut = mut_func(X, gamma=gamma)

            # 如果返回 None 或包含 NaN/Inf
            if K_mut is None or not np.isfinite(K_mut).all():
                vectors.append(np.ones(11) * 999)
                continue

            # 为防止指数溢出，可以 clip
            K_mut = np.clip(K_mut, -1e5, 1e5)

        except Exception:
            vectors.append(np.ones(11) * 999)
            continue

        # 1. 约束违反
        mut_chk = check_constraints(K_mut)

        max_dev = mut_chk["max_deviation"]

        # 6个约束向量（长度固定6）
        constr_vec = [0.0 if mut_chk[key] else max_dev for key in
                      ["type_valid", "shape_valid", "nan_inf_valid",
                       "symmetry_valid", "diag_valid", "range_valid"]]

        # 2. 输出统计特征（长度固定4）
        stat_vec = [np.nanmean(K_mut), np.nanstd(K_mut),
                    np.nanmax(K_mut), np.nanmin(K_mut)]

        # 3. 违反规则占比（长度1）
        viol_vec = [len(mut_chk["violations"])]

        # 总长度固定 6+4+1 = 11
        vec = constr_vec + stat_vec + viol_vec
        vectors.append(vec)

    vectors = np.array(vectors, dtype=float)
    return np.nanmean(vectors, axis=0)


# -----------------------------
# 2. 聚类约简
# -----------------------------
def cluster_reduction(fingerprints, n_clusters=10):
    X = np.array(fingerprints)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 为每个簇挑选代表：这里用第一条出现的作为代表
    representatives = {}
    for idx, label in enumerate(labels):
        if label not in representatives:
            representatives[label] = idx

    return representatives, labels


# -----------------------------
# 3. 主程序
# -----------------------------
def run_fingerprinting_and_clustering():
    tests = generate_lhs_tests(TEST_CASE_NUM, DIM, POINTS)

    # 加载原始程序
    M00 = importlib.import_module("M00")
    ref_func = M00.rbf_kernel

    fingerprints = []
    mutant_names = []

    for i in range(1, 76):
        name = f"M{str(i).zfill(2)}"
        if not os.path.exists(name + ".py"):
            continue

        try:
            module = importlib.import_module(name)
            mut_func = module.rbf_kernel
        except:
            continue

        fp = build_fingerprint(mut_func, ref_func, tests)
        fingerprints.append(fp)
        mutant_names.append(name)

    # 聚类约简
    representatives, labels = cluster_reduction(fingerprints, n_clusters=15)

    print("\n===== Cluster Representatives =====")
    for lbl, idx in representatives.items():
        print(f"Cluster {lbl}: {mutant_names[idx]}")

    return mutant_names, fingerprints, labels, representatives


# -----------------------------
if __name__ == "__main__":
    run_fingerprinting_and_clustering()