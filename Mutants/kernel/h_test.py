import numpy as np
import warnings
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# 忽略数值警告
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# =====================================================
# 1. 约束层定义（6个核心约束）
# =====================================================

class RBFConstraints:
    """RBF核的数学约束检查器"""
    
    @staticmethod
    def check_type(K: Any) -> Tuple[bool, float]:
        """检查类型是否为ndarray"""
        is_valid = isinstance(K, np.ndarray)
        return not is_valid, 0.0 if is_valid else 1.0
    
    @staticmethod
    def check_shape(K: np.ndarray, expected_shape: Tuple) -> Tuple[bool, float]:
        """检查形状是否匹配"""
        if not isinstance(K, np.ndarray):
            return True, 1.0
        is_valid = K.shape == expected_shape
        return not is_valid, 0.0 if is_valid else 1.0
    
    @staticmethod
    def check_numerical_stability(K: np.ndarray) -> Tuple[bool, float]:
        """检查NaN/Inf"""
        if not isinstance(K, np.ndarray):
            return True, 1.0
        has_inf = np.any(np.isinf(K))
        has_nan = np.any(np.isnan(K))
        is_violated = has_inf or has_nan
        # 严重程度：只要有一个就最严重
        severity = 1.0 if is_violated else 0.0
        return is_violated, severity
    
    @staticmethod
    def check_symmetry(K: np.ndarray, is_self_similar: bool) -> Tuple[bool, float]:
        """
        检查对称性
        is_self_similar: True表示X is Y（应该对称）
        """
        if not isinstance(K, np.ndarray) or K.ndim != 2:
            return True, 1.0
        
        if not is_self_similar:
            # 如果不是自相似，不要求严格对称
            return False, 0.0
        
        diff = np.abs(K - K.T)
        max_diff = np.max(diff)
        is_violated = max_diff > 1e-7
        
        # 严重程度归一化（假设最大合理差异为1.0）
        severity = min(max_diff, 1.0)
        return is_violated, severity
    
    @staticmethod
    def check_diag_unity(K: np.ndarray, is_self_similar: bool) -> Tuple[bool, float]:
        """修复版：严格区分 X=Y 和 X≠Y"""
        if not isinstance(K, np.ndarray) or K.ndim != 2:
            return True, 1.0
        
        if not is_self_similar:
            # X≠Y 时，对角线不需要为 1，直接返回"未违反"
            return False, 0.0
        
        # X=Y 时，严格检查对角线为 1
        diag = np.diag(K)
        deviation = np.abs(diag - 1.0)
        max_dev = np.max(deviation)
        is_violated = max_dev > 1e-5
        
        return is_violated, min(max_dev, 1.0)
    
    @staticmethod
    def check_range(K: np.ndarray) -> Tuple[bool, float]:
        """检查值域[0,1]"""
        if not isinstance(K, np.ndarray):
            return True, 1.0
        
        min_val = np.min(K)
        max_val = np.max(K)
        
        # 负值程度（超过容忍度-1e-10）
        neg_violation = abs(min_val) if min_val < -1e-10 else 0.0
        
        # 超上限程度
        upper_violation = max_val - 1.0 if max_val > 1.0 else 0.0
        
        is_violated = (min_val < -1e-10) or (max_val > 1.0)
        severity = max(neg_violation, upper_violation)
        
        return is_violated, severity
    
    @classmethod
    def check_all(cls, K: Any, expected_shape: Tuple, is_self_similar: bool) -> Dict:
        """执行所有约束检查"""
        results = {}
        
        # 1. 类型检查
        violated, sev = cls.check_type(K)
        results['type'] = {
            'violated': violated,
            'severity': sev,
            'label': '类型异常' if violated else '正常'
        }
        
        if violated:
            # 类型错误，后续检查无意义
            return results
        
        # 2. 形状检查
        violated, sev = cls.check_shape(K, expected_shape)
        results['shape'] = {
            'violated': violated,
            'severity': sev,
            'label': '形状异常' if violated else '正常'
        }
        
        if violated:
            return results
        
        # 3. 数值稳定性
        violated, sev = cls.check_numerical_stability(K)
        results['stability'] = {
            'violated': violated,
            'severity': sev,
            'label': '数值溢出' if violated else '正常'
        }
        
        if violated:
            return results
        
        # 4. 对称性（仅自相似时严格）
        violated, sev = cls.check_symmetry(K, is_self_similar)
        results['symmetry'] = {
            'violated': violated,
            'severity': sev,
            'label': '非对称输出' if violated else '正常'
        }
        
        # 5. 对角线（仅自相似时检查）
        violated, sev = cls.check_diag_unity(K, is_self_similar)
        results['diag'] = {
            'violated': violated,
            'severity': sev,
            'label': '对角线偏离' if violated else '正常'
        }
        
        # 6. 值域
        violated, sev = cls.check_range(K)
        results['range'] = {
            'violated': violated,
            'severity': sev,
            'label': '值域异常' if violated else '正常'
        }
        
        return results

# =====================================================
# 2. 分层判定引擎
# =====================================================

class HierarchicalOracle:
    """
    三层判定引擎：
    Layer 1: 数值容差（严格1e-9 vs 宽松1e-4）
    Layer 2: 约束违反记录
    Layer 3: 约束类型集合 + 量级相似度
    """
    
    def __init__(self):
        self.tolerance_strict = 1e-6
        self.tolerance_loose = 1e-2
        self.magnitude_bins = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]  # 数量级分桶
    
    def get_magnitude_level(self, severity: float) -> int:
        """获取量级等级（用于Layer 3比较）"""
        if severity <= 0:
            return -1  # 无违反
        for i, threshold in enumerate(self.magnitude_bins):
            if severity <= threshold:
                return i
        return len(self.magnitude_bins)
    
    def judge(self, original_output: Any, mutant_output: Any, 
              expected_shape: Tuple, is_self_similar: bool) -> Dict:
        """
        执行三层判定
        返回: {
            'is_kill': bool,
            'is_equiv': bool,  # 是否等效（伪杀死）
            'layer': int,      # 在哪一层判定
            'reason': str,
            'details': dict
        }
        """
        # Layer 1: 数值容差初筛
        layer1_result = self._layer1_numerical(original_output, mutant_output)
        
        if layer1_result['strict_equiv']:
            # 严格等效
            return {
                'is_kill': False,
                'is_equiv': True,
                'layer': 1,
                'reason': '严格数值等效（<1e-9）',
                'details': layer1_result
            }
        
        if layer1_result['significant_diff']:
            # 显著差异，直接杀死
            return {
                'is_kill': True,
                'is_equiv': False,
                'layer': 1,
                'reason': f'显著数值差异（>{self.tolerance_loose}）',
                'details': layer1_result
            }
        
        # Layer 2: 约束违反检查（进入模糊区）
        orig_constraints = RBFConstraints.check_all(
            original_output, expected_shape, is_self_similar
        )
        mut_constraints = RBFConstraints.check_all(
            mutant_output, expected_shape, is_self_similar
        )
        
        # 提取违反的约束类型
        orig_violations = {k for k, v in orig_constraints.items() if v['violated']}
        mut_violations = {k for k, v in mut_constraints.items() if v['violated']}
        
        layer2_result = {
            'orig_violations': orig_violations,
            'mut_violations': mut_violations,
            'orig_details': orig_constraints,
            'mut_details': mut_constraints
        }
        
        # 检查Critical约束（类型、形状、稳定性）
        critical = {'type', 'shape', 'stability'}
        orig_critical = orig_violations & critical
        mut_critical = mut_violations & critical
        
        # 如果原始通过而变异体失败，直接杀死
        if not orig_critical and mut_critical:
            return {
                'is_kill': True,
                'is_equiv': False,
                'layer': 2,
                'reason': f'变异体违反关键约束: {mut_critical}',
                'details': layer2_result
            }
        
        # Layer 3: 约束模式相似度（两者都违反或都通过）
        layer3_result = self._layer3_pattern_similarity(
            orig_violations, mut_violations,
            orig_constraints, mut_constraints
        )
        
        if layer3_result['is_equiv']:
            return {
                'is_kill': False,
                'is_equiv': True,
                'layer': 3,
                'reason': layer3_result['reason'],
                'details': {
                    'layer1': layer1_result,
                    'layer2': layer2_result,
                    'layer3': layer3_result
                }
            }
        else:
            return {
                'is_kill': True,
                'is_equiv': False,
                'layer': 3,
                'reason': layer3_result['reason'],
                'details': {
                    'layer1': layer1_result,
                    'layer2': layer2_result,
                    'layer3': layer3_result
                }
            }
    
    def _layer1_numerical(self, orig: Any, mut: Any) -> Dict:
        """Layer 1: 数值容差检查"""
        result = {
            'strict_equiv': False,
            'significant_diff': False,
            'max_diff': float('inf')
        }
        
        # 类型检查
        if type(orig) != type(mut):
            result['significant_diff'] = True
            return result
        
        if not isinstance(orig, np.ndarray):
            # 都不是数组，直接比较
            result['max_diff'] = abs(orig - mut) if isinstance(orig, (int, float)) else 1.0
            result['strict_equiv'] = result['max_diff'] < self.tolerance_strict
            result['significant_diff'] = result['max_diff'] > self.tolerance_loose
            return result
        
        # 都是数组
        if orig.shape != mut.shape:
            result['significant_diff'] = True
            return result
        
        # 计算差异
        diff = np.abs(orig - mut)
        result['max_diff'] = np.max(diff)
        
        result['strict_equiv'] = result['max_diff'] < self.tolerance_strict
        result['significant_diff'] = result['max_diff'] > self.tolerance_loose
        
        return result
    
    def _layer3_pattern_similarity(self, orig_vio: set, mut_vio: set,
                                   orig_det: Dict, mut_det: Dict) -> Dict:
        """Layer 3: 约束模式相似度"""
        
        # 规则1: 约束类型集合完全相同
        if orig_vio != mut_vio:
            return {
                'is_equiv': False,
                'reason': f'约束类型不同: 原始{orig_vio} vs 变异{mut_vio}'
            }
        
        # 如果没有违反任何约束（都通过），但在Layer 1是模糊区（数值微差）
        if not orig_vio:
            return {
                'is_equiv': True,
                'reason': '都满足所有约束，数值微差视为等效'
            }
        
        # 规则2: 最大偏离量级在同一数量级（10倍以内）
        max_orig_sev = max((orig_det[k]['severity'] for k in orig_vio), default=0.0)
        max_mut_sev = max((mut_det[k]['severity'] for k in mut_vio), default=0.0)
        
        # 获取量级等级
        orig_level = self.get_magnitude_level(max_orig_sev)
        mut_level = self.get_magnitude_level(max_mut_sev)
        
        # 检查是否在同一桶或相邻桶（允许1级误差）
        if abs(orig_level - mut_level) <= 1:
            return {
                'is_equiv': True,
                'reason': f'约束类型相同({orig_vio})，量级相近(原始:{max_orig_sev:.2e}, 变异:{max_mut_sev:.2e})'
            }
        else:
            return {
                'is_equiv': False,
                'reason': f'量级差异过大: 原始{max_orig_sev:.2e}(level{orig_level}) vs 变异{max_mut_sev:.2e}(level{mut_level})'
            }

# =====================================================
# 3. 测试用例生成（LHS采样）
# =====================================================

def generate_test_cases(n: int = 50, seed: int = 42) -> List[Tuple]:
    """生成LHS测试用例"""
    np.random.seed(seed)
    test_cases = []
    n_samples, n_features = 8, 4
    
    for i in range(n):
        # LHS采样
        X = np.zeros((n_samples, n_features))
        Y = np.zeros((n_samples, n_features))
        
        for f in range(n_features):
            centers = np.linspace(-5, 5, max(2, n_samples // 2))
            X[:, f] = np.random.choice(centers, n_samples) + np.random.uniform(-0.5, 0.5, n_samples)
            Y[:, f] = np.random.choice(centers, n_samples) + np.random.uniform(-0.5, 0.5, n_samples)
        
        gamma = 10 ** np.random.uniform(-3, 3)
        eps = 10 ** np.random.uniform(-12, -4)
        
        # 混合测试类型
        det = i % 10
        if det < 3:
            test_cases.append((X, None, gamma, eps, True))  # X=Y, 自相似
        elif det < 8:
            test_cases.append((X, X.copy(), gamma, eps, True))  # X=Y (显式)
        else:
            test_cases.append((X, Y, gamma, eps, False))  # X!=Y
    
    # 边界测试
    boundary_cases = [
        (np.zeros((4, 4)), None, 1.0, 1e-8, True),  # 全零
        (np.ones((4, 4)), None, 1.0, 1e-8, True),   # 全一
        (np.random.randn(4, 4) * 1e5, None, 1e-3, 1e-8, True),  # 大数值
    ]
    
    # 插入边界用例
    indices = np.linspace(0, n-1, len(boundary_cases), dtype=int)
    for idx, case in zip(indices, boundary_cases):
        test_cases[idx] = case
    
    K_orig = np.array([[1.0, 0.5], [0.5, 1.0]])
    K_mut = np.array([[1.0, 0.500001], [0.500002, 1.0]])  # 轻微不对称

    return test_cases

# =====================================================
# 4. 变异体加载与执行
# =====================================================

def load_kernel_func(py_file: Path):
    """动态加载rbf_kernel函数"""
    name = py_file.stem
    if name in sys.modules:
        del sys.modules[name]
    
    spec = importlib.util.spec_from_file_location(name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    
    return mod.rbf_kernel

def run_experiment(mutants_dir: Path, n_tests: int = 50):
    """修复版：正确追踪Layer 2/3触发"""
    print("="*60)
    print("RBF核变异体分层判定实验（修复版）")
    print("="*60)
    
    # 加载原始程序
    oracle_path = mutants_dir / "M00.py"
    oracle_func = load_kernel_func(oracle_path)
    
    # 生成测试用例
    test_cases = generate_test_cases(n_tests)
    print(f"\n生成测试用例: {len(test_cases)}个")
    
    # 初始化判定引擎（可以尝试调整阈值）
    oracle_engine = HierarchicalOracle()
    # 临时调整阈值以强制进入Layer 2/3进行验证：
    # oracle_engine.tolerance_strict = 1e-6  # 取消注释以测试
    # oracle_engine.tolerance_loose = 1e-2   # 取消注释以测试
    
    # 统计容器（修复版）
    results = {
        'total_mutants': 0,
        'true_kills': 0,
        'pseudo_kills': 0,
        'layer_reached': {1: 0, 2: 0, 3: 0},  # 变异体"到达过"该层
        'layer_final': {1: 0, 2: 0, 3: 0},    # 变异体"最终判定"在该层
        'layer2_details': [],  # 记录进入Layer 2的具体情况
        'layer3_details': []   # 记录进入Layer 3的具体情况
    }
    
    mutant_files = sorted([f for f in mutants_dir.glob("M[0-9][0-9].py") if f.stem != "M00"])
    print(f"\n发现变异体: {len(mutant_files)}个")
    
    for m_file in mutant_files:
        mid = m_file.stem
        results['total_mutants'] += 1
        
        try:
            mut_func = load_kernel_func(m_file)
        except Exception as e:
            print(f"{mid}: 加载失败")
            continue
        
        # 每个变异体跨所有测试用例的判定结果
        test_results = []
        layers_reached = set()  # 该变异体到达过的所有层
        
        for idx, (X, Y, gamma, eps, is_self) in enumerate(test_cases):
            # 执行
            try:
                orig_out = oracle_func(X, Y, gamma, eps)
            except Exception as e:
                orig_out = None
            
            try:
                mut_out = mut_func(X, Y, gamma, eps)
            except Exception as e:
                mut_out = None
            
            expected_shape = (X.shape[0], Y.shape[0] if Y is not None else X.shape[0])
            
            # 判定
            judgment = oracle_engine.judge(orig_out, mut_out, expected_shape, is_self)
            if judgment['layer'] == 3 and '原始' in judgment.get('reason', ''):
                print(f"\n[调试] {mid} 测试用例 {idx}:")
                print(f"  is_self: {is_self} (X=Y? {Y is X or Y is None})")
                print(f"  gamma={gamma:.2e}, eps={eps:.2e}")
                
                if isinstance(orig_out, np.ndarray):
                    print(f"  原始输出形状: {orig_out.shape}")
                    print(f"  原始输出 min/max: {np.min(orig_out):.2e} / {np.max(orig_out):.2e}")
                    if orig_out.ndim == 2 and orig_out.shape[0] > 0:
                        print(f"  原始对角线前3个: {np.diag(orig_out)[:3]}")
                        print(f"  原始对称误差: {np.max(np.abs(orig_out - orig_out.T)):.2e}")
                
                # 显示Layer 2的详细信息
                layer2_details = judgment['details'].get('layer2', {})
                print(f"  原始违反约束: {layer2_details.get('orig_violations', set())}")
                print(f"  变异违反约束: {layer2_details.get('mut_violations', set())}")

            test_results.append(judgment)
            layers_reached.add(judgment['layer'])
            
            # 记录进入Layer 2/3的具体案例（用于调试）
            if judgment['layer'] == 2:
                results['layer2_details'].append({
                    'mid': mid,
                    'test_idx': idx,
                    'reason': judgment['reason'],
                    'orig_violations': judgment['details'].get('orig_violations', set()),
                    'mut_violations': judgment['details'].get('mut_violations', set())
                })
            elif judgment['layer'] == 3:
                results['layer3_details'].append({
                    'mid': mid,
                    'test_idx': idx,
                    'reason': judgment['reason']
                })
        
        # 汇总该变异体
        kill_count = sum(1 for r in test_results if r['is_kill'])
        equiv_count = sum(1 for r in test_results if r['is_equiv'])
        
        # 更新"到达过"的统计
        for layer in layers_reached:
            results['layer_reached'][layer] += 1
        
        # 最终判定（只要有1个杀死，就算杀死）
        is_killed = kill_count > 0
        is_equiv = equiv_count == len(test_results)
        
        if is_killed:
            results['true_kills'] += 1
        else:
            results['pseudo_kills'] += 1
        
        # 最终判定层：取最深的层（3>2>1）
        final_layer = max(layers_reached)
        results['layer_final'][final_layer] += 1
    
    # 输出结果
    print("\n" + "="*60)
    print("实验结果统计（修复版）")
    print("="*60)
    print(f"总变异体数: {results['total_mutants']}")
    print(f"真实杀死: {results['true_kills']} ({results['true_kills']/results['total_mutants']*100:.1f}%)")
    print(f"伪杀死(等效): {results['pseudo_kills']} ({results['pseudo_kills']/results['total_mutants']*100:.1f}%)")
    
    print(f"\n【关键】到达过各层的变异体数:")
    print(f"  Layer 1 (数值容差): {results['layer_reached'][1]}个")
    print(f"  Layer 2 (约束检查): {results['layer_reached'][2]}个")  
    print(f"  Layer 3 (模式相似): {results['layer_reached'][3]}个")
    
    print(f"\n【最终判定】在各层的变异体数:")
    print(f"  Layer 1: {results['layer_final'][1]}个")
    print(f"  Layer 2: {results['layer_final'][2]}个")
    print(f"  Layer 3: {results['layer_final'][3]}个")
    
    # 显示Layer 2/3的具体案例
    if results['layer2_details']:
        print(f"\n进入Layer 2的测试用例数: {len(results['layer2_details'])}")
        print("示例（前3个）:")
        for detail in results['layer2_details'][:3]:
            print(f"  {detail['mid']}[测试{detail['test_idx']}]: {detail['reason']}")
    
    if results['layer3_details']:
        print(f"\n进入Layer 3的测试用例数: {len(results['layer3_details'])}")
        print("示例（前3个）:")
        for detail in results['layer3_details'][:3]:
            print(f"  {detail['mid']}[测试{detail['test_idx']}]: {detail['reason']}")
    else:
        print(f"\n进入Layer 3的测试用例数: 0")
# =====================================================
# 5. 主入口
# =====================================================

# if __name__ == "__main__":
#     # 假设脚本在变异体目录中运行，或指定目录
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dir', type=str, default='.', help='变异体所在目录')
#     parser.add_argument('--n_tests', type=int, default=50, help='测试用例数')
#     args = parser.parse_args()
    
#     mutants_dir = Path(args.dir)
#     run_experiment(mutants_dir, args.n_tests)

def test_layer2_trigger_fixed():
    """修复版：正确构造 Layer 2 触发场景"""
    print("\n" + "="*60)
    print("强制触发 Layer 2/3 测试（修复版）")
    print("="*60)
    
    oracle = HierarchicalOracle()
    # 调整阈值，让 1e-6 落在模糊区
    oracle.tolerance_strict = 1e-8
    oracle.tolerance_loose = 1e-4

    # 【场景1】两者都轻微不对称（修复：让原始也有轻微不对称）
    print("\n【场景1】两者都轻微不对称（应进入Layer 3）")
    K_orig = np.array([
        [1.0, 0.60653066, 0.13533528],
        [0.60653066, 1.0, 0.60653066],
        [0.13533528, 0.60653066, 1.0]
    ])
    # 给原始也添加轻微不对称（1e-6 级）
    K_orig[0, 1] += 0.5e-6
    K_orig[1, 0] -= 0.5e-6  # 现在原始也是轻微不对称
    
    K_mut = K_orig.copy()
    K_mut[0, 1] += 1e-6  # 不对称程度稍大
    K_mut[1, 0] -= 1e-6
    
    result = oracle.judge(K_orig, K_mut, (3, 3), True)
    print(f"判定结果: Layer {result['layer']}")
    if result['layer'] >= 2:
        layer2 = result['details']['layer2']
        print(f"原始违反: {layer2['orig_violations']}")
        print(f"变异违反: {layer2['mut_violations']}")
        if result['layer'] == 3:
            print(f"Layer 3判定: {result['details']['layer3']['is_equiv']}")

    # 【场景2】两者都轻微超范围（应进入Layer 3且等效）
    print("\n【场景2】两者都轻微超范围（应等效）")
    K_orig2 = np.array([
        [1.0, 1.000001, 0.5],  # 1.000001 轻微超范围
        [1.000001, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])
    K_mut2 = K_orig2.copy()
    K_mut2[0, 1] = 1.000002  # 同样轻微超范围，程度相近
    
    result2 = oracle.judge(K_orig2, K_mut2, (3, 3), True)
    print(f"判定结果: Layer {result2['layer']}, 等效: {result2['is_equiv']}")
    if result2['layer'] >= 2:
        print(f"违反约束: {result2['details']['layer2']['orig_violations']}")

if __name__ == "__main__":
    test_layer2_trigger_fixed()

