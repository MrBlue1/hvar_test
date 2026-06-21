import numpy as np
import importlib.util
from sklearn.cluster import KMeans
from collections import defaultdict
from Mutants.experiment_1 import plot_coverage,plot_cm_both
from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies,plot_cases
from Mutants.experiment_rq3 import run_rq3_experiment
from Mutants.experiment_rq4 import run_rq4_experiment_a,run_rq4_experiment_b
from Mutants.mutant_ananlysis import run_optimized_analysis,detect_fingerprint_collisions2, run_violation_statistc,analyze_single_operator,print_operator_summary,conditional_diversity_test,resolution_gain_test
import warnings
import pickle

np.seterr(invalid='ignore')  # 对inf-nan计算不再报警
# 方案2: 仅忽略RuntimeWarning（更精细）
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
# ==========================
# rbf_kernel 输出可能的分类
# ==========================
all_behavior_types=['invalid_output','negative_values', 'uniform_centrality', 'exceeds_one', 'near_identical', 'nan', 'symmetry_violation', 'singular', 'negative_row_sum', 'non_psd', 'similarity_inversion', 'trace_violation', 'low_rank', 'inf', 'exception', 'homogeneous_offdiag', 'excessive_sparsity', 'expanded_norm', 'diagonal_not_one', 'eigenvalue_failure', 'contracted_norm','never_happen']


# ==========================
# 测试用例生成（LHS增强版）
# ==========================
def generate_lhs_tests(n_samples=200):
    tests = []
    for _ in range(n_samples):
        # 增加负值模式以检测符号敏感变异体（如abs替换）
        dim_type = np.random.choice(['1D', '2D', 'EXTREME', 'NEGATIVE'], p=[0.35, 0.35, 0.2, 0.1])
        
        if dim_type == '1D':
            size = np.random.randint(2, 10)
            X = np.random.randn(size)  # 标准正态分布（含负值）
        elif dim_type == '2D':
            rows, cols = np.random.randint(2, 10, size=2)
            X = np.random.randn(rows, cols)
        elif dim_type == 'NEGATIVE':  # 专门生成强负值
            rows, cols = np.random.randint(2, 6, size=2)
            X = -np.random.rand(rows, cols) * 10  # [-10, 0)
        else:  # EXTREME
            mode = np.random.choice(['large', 'small', 'zero', 'inf', 'nan'])
            shape = (np.random.randint(2, 5), np.random.randint(2, 5))
            if mode == 'large':
                X = np.random.randn(*shape) * 1e200  # 保持符号的极端大数
            elif mode == 'small':
                X = np.random.randn(*shape) * 1e-200
            elif mode == 'zero':
                X = np.zeros(shape)
            elif mode == 'inf':
                X = np.full(shape, np.inf)
                X[np.random.randint(shape[0]), np.random.randint(shape[1])] = np.random.randn()
            else:  # nan
                X = np.random.randn(*shape)
                X[np.random.randint(shape[0]), np.random.randint(shape[1])] = np.nan
        
        # Y生成：匹配X的分布特性，增加符号冲突概率
        if np.random.rand() < 0.7:
            if dim_type in ['EXTREME', 'NEGATIVE']:
                Y = X.copy() if np.random.rand() < 0.3 else None
            else:
                # 随机决定Y是否与X同分布（增加负值组合）
                Y = np.random.randn(*X.shape) if np.random.rand() < 0.5 else None
        else:
            Y = None
            
        tests.append((X, Y))
    return tests

# ==========================
# 载入 Oracle & Mutants
# ==========================
def load_oracle():
    try:
        spec = importlib.util.spec_from_file_location('M00', 'M00.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.rbf_kernel
    except Exception as e:
        print(f"[WARN] Failed to load Oracle: {e}")
        return None

def load_mutants(mutant_files):
    mutant_funcs = {}
    for mf in mutant_files:
        name = mf.split(".")[0]
        try:
            spec = importlib.util.spec_from_file_location(name, mf)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mutant_funcs[name] = mod.rbf_kernel
        except Exception as e:
            print(f"[WARN] Failed to load {mf}: {e}")
    return mutant_funcs

# ==========================
# 三层 Layered Decision Engine
# ==========================
def detect_generic_violations(output):
    violations = []    
    
    if np.isnan(output).any():
        # print(f'isnan: {output}; ')
        # print(f'')    
        violations.append("nan")
    if np.isinf(output).any():
        # print(f'isinf: {output}; ')
        # print(f'')  
        violations.append("inf")
    return violations

def detect_rbf_kernel_violations(K, tol=1e-5):
    violations = detect_generic_violations(K)
    if K is None or not isinstance(K, np.ndarray) or K.ndim != 2:
        return ["invalid_output"] + violations
    
    n = K.shape[0]
    
    # 1. 基本性质
    if not np.allclose(K, K.T, atol=tol):
        violations.append("symmetry_violation")
    if not np.allclose(np.diag(K), 1.0, atol=tol):
        violations.append("diagonal_not_one")
    
    # 2. 范围检查
    if (K < -tol).any(): violations.append("negative_values")
    if (K > 1 + tol).any(): violations.append("exceeds_one")
    
    # 3. 正定性 & 谱分析
    try:
        eigs = np.linalg.eigvalsh(K)
        if (eigs < -tol).any(): violations.append("non_psd")
        if np.min(eigs) < 1e-12: violations.append("singular")
        
        # 有效秩检查（RBF应有完整秩或接近完整）
        rank = np.sum(eigs > tol)
        if rank < n * 0.5: violations.append("low_rank")
        
        # 迹异常（应为n）
        if not np.isclose(np.trace(K), n, atol=tol*n):
            violations.append("trace_violation")
            
    except: violations.append("eigenvalue_failure")
    
    # 4. 矩阵范数异常
    frob_norm = np.linalg.norm(K, 'fro')
    if frob_norm < np.sqrt(n) * 0.9: violations.append("contracted_norm")
    if frob_norm > np.sqrt(n) * n: violations.append("expanded_norm")
    
    # 5. 行和/列和异常（如果是相似度矩阵，行和反映中心性）
    row_sums = K.sum(axis=1)
    if np.any(row_sums < 0): violations.append("negative_row_sum")
    if np.allclose(row_sums, row_sums[0]) and n > 1:
        violations.append("uniform_centrality")  # 异常均匀
    
    # 6. 稀疏性异常（RBF不应稀疏）
    zero_ratio = np.sum(K < tol) / K.size
    if zero_ratio > 0.5: violations.append("excessive_sparsity")
    
    # 7. 距离单调性违反（抽样检查）
    if n > 2:
        samples = np.random.choice(n, min(3, n), replace=False)
        for i in samples:
            for j in samples:
                if i != j:
                    # K[i,i] - K[i,j] 应 >= 0（自相似最大）
                    if K[i,i] < K[i,j] - tol:
                        violations.append("similarity_inversion")
                        break
    
    # 8. 数值精度损失
    if np.any(np.abs(K) < 1e-300) and np.any(np.abs(K) > 1e300):
        violations.append("dynamic_range_collapse")
    
    # 9. 非对角线统计异常
    off_diag = K[~np.eye(n, dtype=bool)]
    if off_diag.size > 0:
        if np.all(off_diag > 0.99): violations.append("near_identical")
        if np.std(off_diag) < 1e-6: violations.append("homogeneous_offdiag")
    

    return sorted(list(set(violations)))

def layered_decision_engine(oracle_out, mutant_out):
    tol = 1e-9
    
    oracle_viol = detect_rbf_kernel_violations(oracle_out, tol)
    mutant_viol = detect_rbf_kernel_violations(mutant_out, tol)

    oracle_has = len(oracle_viol) > 0
    mutant_has = len(mutant_viol) > 0    

    # Layer 3 判定逻辑
    # 1. 都没有违规
    # if not oracle_has and not mutant_has:
    #     return not np.allclose(oracle_out, mutant_out, atol=tol), oracle_viol, mutant_viol
    # # 2. 保证其中一个违规一个不违规
    # if oracle_has^mutant_has:
    #     return True, oracle_viol, mutant_viol
    # # 3. 两者都违规
    # if set(oracle_viol) != set(mutant_viol):       
    #     return True, oracle_viol, mutant_viol
    # return False, oracle_viol, mutant_viol

    return not np.allclose(oracle_out, mutant_out, atol=tol), oracle_viol, mutant_viol

# ==========================
# 运行测试
# ==========================
oracle = load_oracle()
def run_test(func, X_Y_tuple, gamma=1.0):
    X, Y = X_Y_tuple
    old = np.seterr(over='raise')
    K, K_O = [], []
    f_err, o_err = None, None
    
    try:
        try:
            K = func(X, Y, gamma=gamma)
        except FloatingPointError:
            f_err = "inf"
        except Exception:
            f_err = "exception"
        
        if K is None:
            K=[]
            f_err = "exception"
            # print(f'f_err========================{f_err}')

        try:
            K_O = oracle(X, Y, gamma=gamma)
        except FloatingPointError:
            o_err = "inf"
        except Exception:
            o_err = "exception"
        if K_O is None:
            K_O=[]
            o_err = "exception"
            # print(f'o_err========================{f_err}')
        
        # 关键修正：任意一方异常都立即返回，不执行引擎
        if f_err is not None and o_err is not None:
            t=f_err==o_err
            return t, [o_err], [f_err]
        if f_err:
            return True, [], [f_err]
        if o_err:
            return True, [o_err], []
        
        # 只有两者都正常时才执行决策引擎
        killed, o_viol, m_viol = layered_decision_engine(K_O, K)
        return killed, o_viol, m_viol
    finally:       
        np.seterr(**old)

# ==========================
# 构建指纹 & MS & 违规类型覆盖
# ==========================
def build_fingerprints(mutant_funcs, tests):
    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}  # 存每个变异体覆盖的违规类型
    all_viol_types = set() #全部可能的违规类型代码
    
    for name, func in mutant_funcs.items():
        vec = np.zeros(len(all_behavior_types)) #全局行为类型向量
        killed_list = []
        viol_types = set()
        for i,test in enumerate(tests):
            killed, o_viol, m_viol = run_test(func, test)
            killed_list.append(1 if killed else 0)
            viol_types.update(m_viol)
            all_viol_types.update(m_viol)  # 收集和去重所有类型
            
            for viol in m_viol:
                if viol in all_behavior_types:
                    idx=all_behavior_types.index(viol)
                    weight = 1 #+ 0.01 * ((i*idx) / len(tests))  # i 从 0 到 n_tests-1
                    vec[idx]+=weight

        ms_per_mutant[name] = np.mean(killed_list)
        fingerprints[name] = np.array(killed_list, dtype=float)
        violation_map[name] = vec
        # print(f"[INFO] Original MS: {name} killed {np.sum(killed_list)}/{len(tests)} = {ms_per_mutant[name]:.2f}")
        # print(f'输出多样性({len(all_viol_types)})：{vec}')
        # print(f'kill向量：({len(killed_list)})：{killed_list}')
        # print(f"[INFO] Original MS: {name} killed {np.sum(killed_list)}/{len(tests)} = {ms_per_mutant[name]:.2f}")
    return fingerprints, ms_per_mutant, violation_map
 
def save_tests_fingerprints(tests,kill_matrix,violation_map,ms_per_mutant):
    with open('vp.pkl','wb') as f:
        pickle.dump({
            'tests':tests,
            'kill_matrix':kill_matrix,
            'violation_map':violation_map,
            'ms_per_mutant':ms_per_mutant
        },f)

def load_test_fingerprints():
    with open('vp.pkl','rb') as f:
        all=pickle.load(f)
    return all

# ==========================
# 示例运行
# ==========================
if __name__ == "__main__":
    categories = {
        'Numerical Stability':   [0, 4, 5, 7, 13, 14, 17, 19, 20, 21],  # invalid_output, near_identical, nan, singular, inf, exception, expanded_norm, eigenvalue_failure, contracted_norm, never_happen
        'Statistical Moments':   [2, 8, 11, 15],                           # uniform_centrality, negative_row_sum, trace_violation, homogeneous_offdiag
        'Distributional Axiom':  [1, 3, 9, 10],                            # negative_values, exceeds_one, non_psd, similarity_inversion
        'Structural Invariants': [6, 12, 16, 18]                           # symmetry_violation, low_rank, excessive_sparsity, diagonal_not_one
    }
    mutant_files = [f"M{i:02d}.py" for i in range(1, 76)]
    tests = generate_lhs_tests(n_samples=300)
    mutant_funcs = load_mutants(mutant_files)
    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutant_funcs, tests)
    
#region RQ1 experiment CM Metrics
    # plot_cm_both(kill_matrix,violation_map)
    # plot_km_fp_confusion_heatmap(kill_matrix,violation_map)
    # matrix=plot_km_layer_heatmap(kill_matrix,violation_map,categories) #这个有问题
    # print(matrix)
#endregion
    
#region RQ1 significance test
    # a=analyze_single_operator('softmax',kill_matrix,violation_map)
    # print_operator_summary(a)
#endregion
    
#region RQ2:experiment A-C
    # print('RQ2:experiment A: Metrics')
    # v=compute_rq2_metrics(kill_matrix, violation_map,categories)
    # print(v)

    # print('RQ2:experiment B: fingerprint tsne')
    # plot_fingerprint_tsne(violation_map,categories,save_path='rq2\tsne.png')

    print('RQ2:experiment C: 3 Cases ')
    cases=extract_case_studies(kill_matrix,violation_map,categories)
    for c in cases:
        print(f"\nCase: {c['m1']} vs {c['m2']}")
        print(f"  KM pattern: {c['km_pattern'][:5]}... (same class)")
    print(f"  FP({c['m1']}): {c['fp_m1']} → {c['layer_name_m1']} (L{c['dominant_m1']})")
    print(f"  FP({c['m2']}): {c['fp_m2']} → {c['layer_name_m2']} (L{c['dominant_m2']})")    
    plot_cases()
#endregion
   
#region RQ3: experiment
    print('RQ3: experiment')    
    results = run_rq3_experiment(kill_matrix, violation_map, categories)
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

#endregion

#region RQ4 experiment A
    print('RQ4 experiment A: CI Interception Rate Validation')
    print('='*40)
    thresholds = [0.80, 0.85, 0.90, 0.95, 1.00]
    for th in thresholds:
        result = run_rq4_experiment_a(
            kill_matrix, violation_map, categories,
            n_fine=20, survival_rate_threshold=th, debug=True
        )

    print(f"Stage 1 Passed: {result['operator_summary']['stage1_passed']}")
    print(f"Stage 1 Failed: {result['operator_summary']['stage1_failed']}")
    print(f"Stage 2 Intercepted: {result['operator_summary']['stage2_intercepted']}")
    print(f"Stage 2 Clean Pass: {result['operator_summary']['stage2_clean_pass']}")
    print(f"IR: {result['core_metrics']['Interception_Rate_IR']:.2%}")
    print(f"CPR: {result['core_metrics']['Clean_Pass_Rate_CPR']:.2%}")
    plot_coverage(kill_matrix, violation_map)
    print('='*40)
    sr_vals = []
    for n in kill_matrix:
        km = np.asarray(kill_matrix[n])
        sr = np.mean(km == 0)
        sr_vals.append((n, sr))

    # 按存活率降序排列
    sr_sorted = sorted(sr_vals, key=lambda x: x[1], reverse=True)

    print("=== 存活率 Top 35 ===")
    for i, (n, sr) in enumerate(sr_sorted[:35]):
        flag = " >=0.95" if sr >= 0.95 else " >=0.90" if sr >= 0.90 else ""
        print(f"{n}: {sr:.4f}{flag}")

    print(f"\n=== 关键统计 ===")
    print(f"存活率 >= 0.95: {sum(1 for _, sr in sr_vals if sr >= 0.95)}")
    print(f"存活率 >= 0.90: {sum(1 for _, sr in sr_vals if sr >= 0.90)}")
    print(f"存活率 >= 1.00: {sum(1 for _, sr in sr_vals if sr >= 1.00)}")
    print(f"最高存活率: {max(sr for _, sr in sr_vals):.4f}")
#endregion

#region RQ4 experiment B
    # 1. 运行实验 A（固定 threshold，非循环）
    result_a = run_rq4_experiment_a(
        kill_matrix, violation_map, categories,
        n_fine=20, 
        survival_rate_threshold=0.90,   # Softmax / LayerNorm 用 0.90
        debug=False                       # 关闭调试输出
    )


    # 2. 提取 intercepted 变异体列表
    intercepted_mutants = result_a['intercepted_analysis']['intercepted_ids']

    print(f"实验 A 拦截变异体数: {len(intercepted_mutants)}")
    print(f"示例 ID: {intercepted_mutants[:5]}")

    # 3. 直接传入实验 B
    result_b = run_rq4_experiment_b(
        intercepted_mutants=intercepted_mutants,
        violation_map=violation_map,
        categories=categories,
        n_fine=20
    )

    # 4. 打印实验 B 核心指标
    print(f"样本量: {result_b['sample_size']}")
    print(f"DSC_KM={result_b['granularity']['DSC_KM']}")
    print(f"DSC_FP_strict={result_b['granularity']['DSC_FP_strict']}")
    print(f"DSC_FP_binned={result_b['granularity']['DSC_FP_binned']}")
    print(f"MLCR={result_b['granularity']['MLCR']}")
    print(f"DE_FP={result_b['diagnostic_entropy']['DE_FP_raw']:.3f} bits")
    print(f"DE_normalized={result_b['diagnostic_entropy']['DE_FP_normalized']:.3f}")
    print(f"Entropy gain={result_b['diagnostic_entropy']['entropy_gain']:.3f} bits")

    for case in result_b['case_reports']:
        print(f"\n--- 案例: {case['mutant_1']} vs {case['mutant_2']} ---")
        print(f"主导层: {case['dominant_layer']}")
        print(f"Kill-Matrix: {case['km_diagnosis']}")
        print(f"指纹差异: L1距离={case['l1_distance']}")
        print(f"  {case['mutant_1']}: {case['fp_insight_m1']}")
        print(f"  {case['mutant_2']}: {case['fp_insight_m2']}")
#endregion


