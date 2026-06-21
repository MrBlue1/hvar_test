import numpy as np
import sys
import importlib.util
from pathlib import Path

# 配置
mutants_dir = Path(__file__).parent
n_tests = 100
seed = 42
atol = 1e-6  # 严格容差

def load_kernel(py_file):
    """加载内核函数"""
    name = py_file.stem
    if name in sys.modules:
        del sys.modules[name]
    
    spec = importlib.util.spec_from_file_location(name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.rbf_kernel

def generate_tests(n, seed=42):
    """生成测试用例"""
    np.random.seed(seed)
    tests = []
    for i in range(n):
        X = np.random.randn(8, 4)
        Y = np.random.randn(8, 4) if i % 3 == 0 else None
        gamma = 10 ** np.random.uniform(-3, 3)
        eps = 1e-8
        tests.append((X, Y, gamma, eps))
    # 添加边界测试
    tests.append((np.full((8, 4), 1e154), None, 1.0, 1e-8))
    tests.append((np.zeros((8, 4)), None, 1.0, 1e-8))
    return tests

def compare_outputs():
    """比较 M00 和 M61 的输出"""
    print("=" * 60)
    print("M00 vs M61 等价性验证")
    print("=" * 60)
    
    # 加载两个内核
    try:
        m00 = load_kernel(mutants_dir / "M00.py")
        m61 = load_kernel(mutants_dir / "M63.py")
        print(f"✓ 成功加载 M00 和 M61")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return
    
    # 生成测试用例
    tests = generate_tests(n_tests, seed)
    print(f"✓ 生成 {len(tests)} 个测试用例")
    
    # 逐个比较
    diff_count = 0
    first_diff = None
    
    for idx, (X, Y, gamma, eps) in enumerate(tests):
        try:
            ref = m00(X, Y, gamma, eps)
            mut = m61(X, Y, gamma, eps)
        except Exception as e:
            print(f"\n测试 {idx}: 异常 - {e}")
            diff_count += 1
            continue
        
        # 检查是否等价
        if ref.shape != mut.shape:
            print(f"\n测试 {idx}: 形状不同 - ref{ref.shape} vs mut{mut.shape}")
            diff_count += 1
            if first_diff is None:
                first_diff = (idx, "shape", ref, mut, X, Y)
            continue
        
        # 数值比较
        if not np.allclose(ref, mut, atol=atol, rtol=1e-5, equal_nan=True):
            diff = np.abs(ref - mut)
            max_diff = np.max(diff)
            diff_loc = np.unravel_index(np.argmax(diff), diff.shape)
            
            print(f"\n测试 {idx}: 数值差异")
            print(f"  最大差异: {max_diff:.2e} @ {diff_loc}")
            print(f"  M00 值: {ref[diff_loc]:.10e}")
            print(f"  M61 值: {mut[diff_loc]:.10e}")
            print(f"  NaN数量: M00={np.sum(np.isnan(ref))}, M61={np.sum(np.isnan(mut))}")
            print(f"  Inf数量: M00={np.sum(np.isinf(ref))}, M61={np.sum(np.isinf(mut))}")
            
            diff_count += 1
            if first_diff is None:
                first_diff = (idx, "value", ref, mut, X, Y)
        else:
            # 完全等价
            pass
    
    # 结果汇总
    print("\n" + "=" * 60)
    if diff_count == 0:
        print("✓ 结论: M61 与 M00 完全等价")
        print(f"  通过 {len(tests)} 个测试用例的验证")
    else:
        print(f"✗ 结论: 发现 {diff_count}/{len(tests)} 个测试用例存在差异")
        if first_diff:
            idx, diff_type, ref, mut, X, Y = first_diff
            print(f"\n  首个差异出现在测试 {idx}")
            if diff_type == "value":
                print(f"  建议检查: M61.py 是否真的只修改了变量名？")
                print(f"  可能原因: 运算符被修改 (如 **2 改为 *X)")
    
    print("=" * 60)

if __name__ == "__main__":
    compare_outputs()