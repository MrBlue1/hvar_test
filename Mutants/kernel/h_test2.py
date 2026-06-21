"""
RBF Kernel Mutant Behavior Evaluation Framework
Core logic validation version

Implements:
1. 6 mathematical constraints
2. 3-layer decision engine
3. LHS test case generation
4. Automatic mutant loading
"""

import os
import importlib
import numpy as np
import traceback
from scipy.stats import qmc


# ==========================================================
# Config
# ==========================================================

STRICT_TOL = 1e-9
LOOSE_TOL = 1e-4

CONSTRAINT_TOL = 1e-10

TEST_CASE_NUM = 30
DIM = 4
POINTS = 6


# ==========================================================
# LHS Test Generator
# ==========================================================

def generate_lhs_tests(num_cases=30, dim=4, points=6):
    sampler = qmc.LatinHypercube(d=dim)
    tests = []

    for _ in range(num_cases):
        X = sampler.random(n=points)
        X = qmc.scale(X, -5, 5)
        gamma = np.random.uniform(0.1, 5.0)
        tests.append((X, gamma))

    return tests


# ==========================================================
# Constraint Layer (6 constraints)
# ==========================================================

def check_constraints(K):

    result = {
        "type_valid": True,
        "shape_valid": True,
        "nan_inf_valid": True,
        "symmetry_valid": True,
        "diag_valid": True,
        "range_valid": True,
        "violations": {},
        "max_deviation": 0.0
    }

    # 1. type check
    if not isinstance(K, np.ndarray):
        result["type_valid"] = False
        result["violations"]["type"] = 1.0
        return result

    # 2. shape check
    if len(K.shape) != 2 or K.shape[0] != K.shape[1]:
        result["shape_valid"] = False
        result["violations"]["shape"] = 1.0
        return result

    # 3. NaN / Inf
    if not np.isfinite(K).all():
        result["nan_inf_valid"] = False
        result["violations"]["nan_inf"] = 1.0

    # 4. symmetry
    sym_diff = np.abs(K - K.T).max()
    if sym_diff > CONSTRAINT_TOL:
        result["symmetry_valid"] = False
        result["violations"]["symmetry"] = sym_diff
        result["max_deviation"] = max(result["max_deviation"], sym_diff)

    # 5. diagonal = 1
    diag_diff = np.abs(np.diag(K) - 1.0).max()
    if diag_diff > CONSTRAINT_TOL:
        result["diag_valid"] = False
        result["violations"]["diag"] = diag_diff
        result["max_deviation"] = max(result["max_deviation"], diag_diff)

    # 6. range [0,1]
    min_val = K.min()
    max_val = K.max()
    range_dev = 0.0

    if min_val < -CONSTRAINT_TOL:
        result["range_valid"] = False
        range_dev = abs(min_val)

    if max_val > 1 + CONSTRAINT_TOL:
        result["range_valid"] = False
        range_dev = max(range_dev, abs(max_val - 1))

    if range_dev > 0:
        result["violations"]["range"] = range_dev
        result["max_deviation"] = max(result["max_deviation"], range_dev)

    return result


# ==========================================================
# Layered Decision Engine
# ==========================================================

def compare_matrices(K_ref, K_mut):

    # Layer 1
    diff = np.abs(K_ref - K_mut).max()

    if diff < STRICT_TOL:
        return "EQUIVALENT_STRICT"

    if diff > LOOSE_TOL:
        return "DIFFERENT"

    # Layer 2
    ref_check = check_constraints(K_ref)
    mut_check = check_constraints(K_mut)

    ref_viol = set(ref_check["violations"].keys())
    mut_viol = set(mut_check["violations"].keys())

    # 如果其中一个违反而另一个没违反
    if ref_viol != mut_viol:
        return "DIFFERENT_CONSTRAINT"

    # Layer 3
    if ref_viol == mut_viol:

        if len(ref_viol) == 0:
            return "EQUIVALENT_NUMERIC_NOISE"

        ref_dev = ref_check["max_deviation"]
        mut_dev = mut_check["max_deviation"]

        # 判断是否同数量级
        if ref_dev == 0 or mut_dev == 0:
            return "DIFFERENT_PATTERN"

        ratio = abs(np.log10(ref_dev) - np.log10(mut_dev))

        if ratio <= 1:  # 同数量级
            return "EQUIVALENT_CONSTRAINT_PATTERN"
        else:
            return "DIFFERENT_PATTERN"

    return "DIFFERENT"


# ==========================================================
# Run Experiment
# ==========================================================

def run_experiment():

    tests = generate_lhs_tests(TEST_CASE_NUM, DIM, POINTS)

    # load original
    M00 = importlib.import_module("M00")
    ref_func = M00.rbf_kernel

    results = {}

    for i in range(1, 76):

        name = f"M{str(i).zfill(2)}"

        if not os.path.exists(name + ".py"):
            continue

        try:
            module = importlib.import_module(name)
            mut_func = module.rbf_kernel
        except:
            continue

        mutant_status = []

        for X, gamma in tests:
            try:
                K_ref = ref_func(X, gamma=gamma)
                K_mut = mut_func(X, gamma=gamma)
                status = compare_matrices(K_ref, K_mut)
                mutant_status.append(status)
            except Exception:
                mutant_status.append("CRASH")

        results[name] = mutant_status

    return results


# ==========================================================
# Summary
# ==========================================================

def summarize(results):

    summary = {}

    for m, statuses in results.items():

        counts = {}
        for s in statuses:
            counts[s] = counts.get(s, 0) + 1

        summary[m] = counts

    return summary


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    results = run_experiment()
    summary = summarize(results)

    print("\n===== Mutant Behavior Summary =====\n")

    for m, stat in summary.items():
        print(m, stat)