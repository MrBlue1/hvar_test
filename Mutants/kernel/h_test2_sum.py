import numpy as np
import importlib.util
from sklearn.cluster import KMeans
import sys

# ==========================
# 1️⃣ 测试用例生成（LHS增强版）
# ==========================
def generate_lhs_tests(n_samples=100):
    tests = []
    for _ in range(n_samples):
        rows_x = np.random.randint(2, 10)
        rows_y = np.random.randint(2, 10)
        dim = np.random.randint(2, 6)

        X = np.random.rand(rows_x, dim)
        Y = np.random.rand(rows_y, dim)
        tests.append((X, Y))
    return tests

# ==========================
# 2️⃣ 动态加载模块
# ==========================
def load_module(module_name, file_path):
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_mutants(mutant_files):
    mutant_funcs = {}
    for mf in mutant_files:
        name = mf.split(".")[0]
        try:
            mod = load_module(name, mf)
            mutant_funcs[name] = mod.rbf_kernel
        except Exception as e:
            print(f"[WARN] Failed to load {mf}: {e}")
    return mutant_funcs

def load_oracle():
    try:
        mod = load_module("M00", "M00.py")
        return mod.rbf_kernel
    except Exception as e:
        print(f"[WARN] Failed to load Oracle: {e}")
        return None

# ==========================
# 3️⃣ 违规检测
# ==========================
def detect_generic_violations(output):
    violations = []
    if not isinstance(output, np.ndarray):
        return ["not_array"]
    if output.ndim != 2:
        return ["dimension_not_2"]
    if np.isnan(output).any():
        violations.append("nan")
    if np.isinf(output).any():
        violations.append("inf")
    return violations

def detect_rbf_kernel_violations(K, X, Y, tol=1e-9):
    violations = detect_generic_violations(K)
    if violations:
        return violations
    # 值域
    if (K <= 0).any() or (K > 1 + tol).any():
        violations.append("out_of_range")
    # 对称与对角线检查，只在 X==Y 时
    if np.array_equal(X, Y) and K.shape[0] == K.shape[1]:
        if not np.allclose(K, K.T, atol=tol):
            violations.append("symmetry_violation")
        if not np.allclose(np.diag(K), 1.0, atol=tol):
            violations.append("diagonal_not_one")
    return sorted(set(violations))

# ==========================
# 4️⃣ Layered Decision Engine
# ==========================
def layered_decision_engine(oracle_out, mutant_out, X, Y, tol=1e-9):
    # Layer 0: 安全检查
    if not isinstance(mutant_out, np.ndarray) or not isinstance(oracle_out, np.ndarray):
        return True, ["not_array"], ["not_array"]
    if oracle_out.shape != mutant_out.shape:
        return True, ["shape_mismatch"], ["shape_mismatch"]

    # Layer 2: 检测违规
    oracle_viol = detect_rbf_kernel_violations(oracle_out, X, Y, tol)
    mutant_viol = detect_rbf_kernel_violations(mutant_out, X, Y, tol)

    oracle_has = len(oracle_viol) > 0
    mutant_has = len(mutant_viol) > 0

    # Layer 3: 判定规则
    # 情况1：都无违规 → 比较数值
    if not oracle_has and not mutant_has:
        if not np.allclose(oracle_out, mutant_out, atol=tol):
            return True, oracle_viol, mutant_viol
        return False, oracle_viol, mutant_viol
    # 情况2：一个违规一个不违规
    if oracle_has != mutant_has:
        return True, oracle_viol, mutant_viol
    # 情况3：都违规，类型不同算杀死
    if set(oracle_viol) != set(mutant_viol):
        return True, oracle_viol, mutant_viol

    # 否则存活
    return False, oracle_viol, mutant_viol

# ==========================
# 5️⃣ 测试运行
# ==========================
oracle = load_oracle()

def run_test(func, X_Y_tuple, gamma=1.0):
    X, Y = X_Y_tuple
    try:
        K_mut = func(X, Y, gamma=gamma)
    except Exception:
        return True
    try:
        K_oracle = oracle(X, Y, gamma=gamma)
    except Exception:
        return False
    killed, _, _ = layered_decision_engine(K_oracle, K_mut, X, Y)
    return killed

# ==========================
# 6️⃣ 构建指纹与 MS
# ==========================
def build_fingerprints(mutant_funcs, tests):
    fingerprints = {}
    ms_per_mutant = {}
    for name, func in mutant_funcs.items():
        killed_list = [run_test(func, test) for test in tests]
        ms = np.mean(killed_list)
        ms_per_mutant[name] = ms
        fingerprints[name] = np.array(killed_list, dtype=float)
        print(f"[INFO] {name}: killed {np.sum(killed_list)}/{len(tests)} = {ms:.2f}")
    return fingerprints, ms_per_mutant

# ==========================
# 7️⃣ 聚类约简
# ==========================
def cluster_reduce(fingerprints, n_clusters=15):
    names = list(fingerprints.keys())
    X = np.array([fingerprints[n] for n in names])
    km = KMeans(n_clusters=min(n_clusters, len(names)), random_state=42)
    labels = km.fit_predict(X)
    cluster_map = {}
    for name, label in zip(names, labels):
        cluster_map.setdefault(label, []).append(name)
    # 选择簇代表：簇内 MS 最大
    representatives = []
    for label, members in cluster_map.items():
        ms_values = [np.mean(fingerprints[m]) for m in members]
        idx = np.argmax(ms_values)
        representatives.append(members[idx])
        print(f"[INFO] Cluster {label} → Rep {members[idx]} | Members {members}")
    return labels, cluster_map, representatives

# ==========================
# 8️⃣ 行为覆盖率与 MS 保留
# ==========================
def compute_coverage_and_retention(ms_per_mutant, fingerprints, representatives, cluster_map):
    # 原始行为覆盖率
    orig_covered = [name for name, ms in ms_per_mutant.items() if ms > 0]
    bc_orig = len(orig_covered) / len(ms_per_mutant) * 100
    # Reduced 行为覆盖率
    reduced_covered = []
    for rep in representatives:
        members = next((m for k, m in cluster_map.items() if rep in m), [])
        if members and np.mean([ms_per_mutant[m] for m in members]) > 0:
            reduced_covered.append(rep)
    bc_reduced = len(reduced_covered) / len(representatives) * 100
    # MS retention
    total_orig_ms = np.mean(list(ms_per_mutant.values()))
    total_reduced_ms = np.mean([np.mean([ms_per_mutant[m] for m in cluster_map[label]]) 
                                for label in cluster_map])
    retention = total_reduced_ms / total_orig_ms * 100.0
    print(f"[INFO] Behavior Coverage Original: {bc_orig:.2f}% | Reduced: {bc_reduced:.2f}%")
    print(f"[INFO] MS Retention: {retention:.2f}%")
    return bc_orig, bc_reduced, retention

# ==========================
# 9️⃣ 主流程
# ==========================
def main():
    mutant_files = [f"M{i:02d}.py" for i in range(1, 76)]
    tests = generate_lhs_tests(n_samples=100)
    mutant_funcs = load_mutants(mutant_files)
    fingerprints, ms_per_mutant = build_fingerprints(mutant_funcs, tests)
    labels, cluster_map, representatives = cluster_reduce(fingerprints, n_clusters=15)
    bc_orig, bc_red, retention = compute_coverage_and_retention(ms_per_mutant, fingerprints, representatives, cluster_map)

if __name__ == "__main__":
    main()