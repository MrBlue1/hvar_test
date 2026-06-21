import sys
import importlib.util

def load_oracle():
    mutant_funcs = {}    
    name = 'M00.py'
    try:
        spec = importlib.util.spec_from_file_location('M00','M00.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mutant_funcs[name] = mod.rbf_kernel
    except Exception as e:
        print(f"[WARN] Failed to load {e}")
    return mutant_funcs

print(load_oracle())


# ==========================
# 7️⃣ 聚类约简
# ==========================
def cluster_reduce(fingerprints, n_clusters=15):
    names = list(fingerprints.keys())
    X = np.array([fingerprints[n] for n in names])

    # 自动调整簇数，避免超过不同指纹数量
    unique_count = len(np.unique(X, axis=0))
    n_clusters = min(n_clusters, unique_count)
    if n_clusters < 1:
        n_clusters = 1

    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)

    cluster_map = {}
    for name, label in zip(names, labels):
        cluster_map.setdefault(label, []).append(name)
    
    # 选择簇代表：簇内 MS 最高者
    representatives = []
    for label, members in cluster_map.items():
        ms_values = [np.mean(fingerprints[m]) for m in members]
        idx = np.argmax(ms_values)
        representatives.append(members[idx])
        print(f"[INFO] Cluster Rep {members[idx]}: members {members}, cluster MS = {ms_values[idx]:.2f}")
    
    return labels, cluster_map, representatives
