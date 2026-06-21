"""
Layer Normalization Mutant Reduction Experiment
Support M00-M120 (121 mutants) generated previously
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import qmc
import warnings
import csv
import os
import sys
import importlib.util
warnings.filterwarnings('ignore')

# ==================== 1. Dynamic Mutant Loader ====================

class MutantLoader:
    """Load mutants from generated py files (M00.py - M120.py)"""
    
    OPERATOR_MAPPING = {
        'M00': 'ORIG',
        # ROR M01-M12
        'M01': 'ROR', 'M02': 'ROR', 'M03': 'ROR', 'M04': 'ROR',
        'M05': 'ROR', 'M06': 'ROR', 'M07': 'ROR', 'M08': 'ROR',
        'M09': 'ROR', 'M10': 'ROR', 'M11': 'ROR', 'M12': 'ROR',
        # AOR M13-M22
        'M13': 'AOR', 'M14': 'AOR', 'M15': 'AOR', 'M16': 'AOR',
        'M17': 'AOR', 'M18': 'AOR', 'M19': 'AOR', 'M20': 'AOR',
        'M21': 'AOR', 'M22': 'AOR',
        # LOR M23-M30
        'M23': 'LOR', 'M24': 'LOR', 'M25': 'LOR', 'M26': 'LOR',
        'M27': 'LOR', 'M28': 'LOR', 'M29': 'LOR', 'M30': 'LOR',
        # COR M31-M36
        'M31': 'COR', 'M32': 'COR', 'M33': 'COR', 'M34': 'COR',
        'M35': 'COR', 'M36': 'COR',
        # UOI M37-M43
        'M37': 'UOI', 'M38': 'UOI', 'M39': 'UOI', 'M40': 'UOI',
        'M41': 'UOI', 'M42': 'UOI', 'M43': 'UOI',
        # SDL M44-M50
        'M44': 'SDL', 'M45': 'SDL', 'M46': 'SDL', 'M47': 'SDL',
        'M48': 'SDL', 'M49': 'SDL', 'M50': 'SDL',
        # ABS M51-M60
        'M51': 'ABS', 'M52': 'ABS', 'M53': 'ABS', 'M54': 'ABS',
        'M55': 'ABS', 'M56': 'ABS', 'M57': 'ABS', 'M58': 'ABS',
        'M59': 'ABS', 'M60': 'ABS',
        # EQUI M61-M75
        'M61': 'EQUI', 'M62': 'EQUI', 'M63': 'EQUI', 'M64': 'EQUI',
        'M65': 'EQUI', 'M66': 'EQUI', 'M67': 'EQUI', 'M68': 'EQUI',
        'M69': 'EQUI', 'M70': 'EQUI', 'M71': 'EQUI', 'M72': 'EQUI',
        'M73': 'EQUI', 'M74': 'EQUI', 'M75': 'EQUI',
        # Extended ROR M76-M85
        'M76': 'ROR', 'M77': 'ROR', 'M78': 'ROR', 'M79': 'ROR',
        'M80': 'ROR', 'M81': 'ROR', 'M82': 'ROR', 'M83': 'ROR',
        'M84': 'ROR', 'M85': 'ROR',
        # Extended AOR M86-M95
        'M86': 'AOR', 'M87': 'AOR', 'M88': 'AOR', 'M89': 'AOR',
        'M90': 'AOR', 'M91': 'AOR', 'M92': 'AOR', 'M93': 'AOR',
        'M94': 'AOR', 'M95': 'AOR',
        # Extended LOR M96-M100
        'M96': 'LOR', 'M97': 'LOR', 'M98': 'LOR', 'M99': 'LOR',
        'M100': 'LOR',
        # Extended COR M101-M105
        'M101': 'COR', 'M102': 'COR', 'M103': 'COR', 'M104': 'COR',
        'M105': 'COR',
        # Extended UOI M106-M110
        'M106': 'UOI', 'M107': 'UOI', 'M108': 'UOI', 'M109': 'UOI',
        'M110': 'UOI',
        # Extended SDL M111-M115
        'M111': 'SDL', 'M112': 'SDL', 'M113': 'SDL', 'M114': 'SDL',
        'M115': 'SDL',
        # Extended ABS M116-M120
        'M116': 'ABS', 'M117': 'ABS', 'M118': 'ABS', 'M119': 'ABS',
        'M120': 'ABS',
    }
    
    def __init__(self, mutants_dir="mutants"):
        self.mutants_dir = mutants_dir
        self.mutant_classes = {}
        self._load_all_mutants()
    
    def _load_all_mutants(self):
        """Dynamically load all mutant files M00-M120"""
        print(f"Loading mutants from {self.mutants_dir}...")
        
        for i in range(121):  # M00 to M120
            mid = f"M{i:02d}" if i < 100 else f"M{i}"  # M00-M99, M100-M120
            filename = f"{mid}.py"
            filepath = os.path.join(self.mutants_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    # Dynamic import
                    spec = importlib.util.spec_from_file_location(f"mutant_{mid}", filepath)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"mutant_{mid}"] = module
                    spec.loader.exec_module(module)
                    
                    # Get LayerNorm class
                    if hasattr(module, 'LayerNorm'):
                        self.mutant_classes[mid] = module.LayerNorm
                    else:
                        print(f"  Warning: {filename} has no LayerNorm class")
                        
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
            else:
                print(f"  Warning: {filename} not found")
        
        print(f"Loaded {len(self.mutant_classes)} mutants: {sorted(self.mutant_classes.keys())[:5]}...{sorted(self.mutant_classes.keys())[-5:]}")
    
    def get_mutant(self, mid):
        """Get mutant class by ID"""
        return self.mutant_classes.get(mid)
    
    def get_operator_type(self, mid):
        """Get operator type for mutant"""
        return self.OPERATOR_MAPPING.get(mid, "UNKNOWN")
    
    def get_all_ids(self):
        """Get all loaded mutant IDs"""
        return sorted(self.mutant_classes.keys())
    
    def get_description(self, mid):
        """Get description from file docstring or mapping"""
        cls = self.get_mutant(mid)
        if cls and cls.__doc__:
            return cls.__doc__.strip()
        return f"{self.get_operator_type(mid)} variant"

# ==================== 2. Test Generation ====================

class TestHarness:
    def __init__(self, n_tests=50, seed=42):
        self.n_tests = n_tests
        self.seed = seed
        self.test_cases = self._generate_lhs_tests()
        self.oracle_outputs = self._compute_oracle()
    
    def _generate_lhs_tests(self):
        """Latin Hypercube Sampling for test case generation"""
        sampler = qmc.LatinHypercube(d=3, seed=self.seed)
        samples = sampler.random(n=self.n_tests)
        
        # Map to realistic shapes: (batch, seq_len, features=64)
        shapes = [(2, 16, 64), (4, 32, 64), (8, 64, 64), (16, 8, 64)]
        test_cases = []
        
        np.random.seed(self.seed)
        for i, sample in enumerate(samples):
            shape_idx = int(sample[0] * len(shapes)) % len(shapes)
            batch, seq, feat = shapes[shape_idx]
            
            # Generate data with different characteristics
            base = np.random.randn(batch, seq, feat).astype(np.float32)
            
            # Edge cases based on sample
            if sample[1] < 0.1:  # Near-zero variance
                base = np.ones((batch, seq, feat)) * np.random.randn() + np.random.randn(batch, seq, feat) * 0.001
            elif sample[1] < 0.2:  # Large magnitude
                base *= 100.0
            elif sample[1] < 0.3:  # Mixed small/large
                base[::2] *= 0.01
                base[1::2] *= 100
            
            test_cases.append(torch.tensor(base))
        
        return test_cases
    
    def _compute_oracle(self):
        """Compute oracle outputs using M00 (original)"""
        if not os.path.exists(os.path.join("mutants", "M00.py")):
            raise FileNotFoundError("M00.py (original) not found in mutants directory")
        
        # Load M00 dynamically
        spec = importlib.util.spec_from_file_location("mutant_M00", os.path.join("mutants", "M00.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["mutant_M00"] = module
        spec.loader.exec_module(module)
        
        sut = module.LayerNorm(64)
        sut.eval()
        with torch.no_grad():
            return [sut(x).numpy() for x in self.test_cases]
    
    def kill_mutant(self, mutant_class):
        """Strong oracle: Check if mutant produces different output"""
        mutant = mutant_class(64)
        mutant.eval()
        with torch.no_grad():
            try:
                mutant_outputs = [mutant(x).numpy() for x in self.test_cases]
                
                kill_vector = []
                for orig, mut in zip(self.oracle_outputs, mutant_outputs):
                    # Check for NaN/Inf
                    if not np.all(np.isfinite(mut)):
                        kill_vector.append(1)
                        continue
                    
                    # Relative error check
                    diff = np.abs(orig - mut)
                    rel_diff = diff / (np.abs(orig) + 1e-8)
                    max_rel_diff = np.max(rel_diff)
                    
                    # Threshold for numerical precision
                    kill_vector.append(1 if max_rel_diff > 1e-4 else 0)
                
                return np.array(kill_vector)
            except Exception as e:
                # Runtime error = killed
                return np.ones(self.n_tests, dtype=int)

# ==================== 3. Behavioral Profiling ====================

class BehavioralProfiler:
    def __init__(self, harness, mutant_loader):
        self.harness = harness
        self.loader = mutant_loader
    
    def profile(self, mid):
        """Extract behavioral fingerprint for mutant"""
        mutant_class = self.loader.get_mutant(mid)
        if mutant_class is None:
            return None
        
        kill_vector = self.harness.kill_mutant(mutant_class)
        
        # Additional behavioral features
        mutant = mutant_class(64)
        mutant.eval()
        with torch.no_grad():
            try:
                outputs = [mutant(x).numpy() for x in self.harness.test_cases]
                stats = {
                    'mean_mean': np.mean([o.mean() for o in outputs]),
                    'mean_std': np.std([o.mean() for o in outputs]),
                    'nan_ratio': np.mean([np.isnan(o).mean() for o in outputs]),
                    'inf_ratio': np.mean([np.isinf(o).mean() for o in outputs]),
                    'max_abs': np.max([np.abs(o).max() for o in outputs]),
                }
            except:
                stats = {
                    'mean_mean': 0, 'mean_std': 0,
                    'nan_ratio': 1.0, 'inf_ratio': 1.0,
                    'max_abs': float('inf')
                }
        
        op_type = self.loader.get_operator_type(mid)
        
        return {
            'id': mid,
            'operator': op_type,
            'description': self.loader.get_description(mid),
            'kill_vector': kill_vector,
            'killed_count': int(np.sum(kill_vector)),
            'behavior_type': self._classify_behavior(kill_vector, stats, op_type),
            'stats': stats,
            'fingerprint': np.concatenate([
                kill_vector,
                [stats['nan_ratio'], stats['inf_ratio'], stats['mean_std']]
            ])
        }
    
    def _classify_behavior(self, kill_vector, stats, op_type):
        """Classify mutant behavior type"""
        if stats['nan_ratio'] > 0.5:
            return "数值溢出"
        elif stats['inf_ratio'] > 0.5:
            return "数值溢出"
        elif np.sum(kill_vector) == 0:
            if op_type == 'EQUI':
                return "等价变异体"
            return "难杀变异体"
        elif np.sum(kill_vector) < len(kill_vector) * 0.3:
            return "边界错误"
        elif stats['max_abs'] > 1000:
            return "超范围错误"
        else:
            return "形状错误"

# ==================== 4. Clustering-based Reduction ====================

class OperatorPriorityReducer:
    """算子优先聚类策略"""
    
    def __init__(self, profiles, budget_k):
        self.profiles = [p for p in profiles if p is not None]
        self.budget_k = budget_k
        self.operators = list(set(p['operator'] for p in self.profiles))
    
    def reduce(self):
        """Operator-priority clustering"""
        # Group by operator
        op_groups = {}
        for p in self.profiles:
            op = p['operator']
            if op not in op_groups:
                op_groups[op] = []
            op_groups[op].append(p)
        
        n_operators = len(op_groups)
        print(f"算子类型数:{n_operators}类, 预算k={self.budget_k}")
        
        # Check budget
        if self.budget_k < n_operators:
            print(f"⚠️ 预算紧张(k={self.budget_k} < 算子数{n_operators})，只覆盖{self.budget_k}类")
            # Select top-k operators by max kill count
            op_kill_counts = {
                op: max(p['killed_count'] for p in group)
                for op, group in op_groups.items()
            }
            selected_ops = sorted(op_kill_counts.keys(), 
                                key=lambda x: op_kill_counts[x], 
                                reverse=True)[:self.budget_k]
        else:
            print(f"✓ 预算充足(k={self.budget_k} >= 算子数{n_operators})，可覆盖所有算子类型")
            selected_ops = list(op_groups.keys())
        
        selected = []
        covered_ops = set()
        
        # First pass: select best from each operator
        for op in selected_ops:
            group = op_groups[op]
            # Select highest kill count as representative
            best = max(group, key=lambda x: x['killed_count'])
            selected.append(best)
            covered_ops.add(op)
            print(f"  选代表: {best['id']} | 算子:{op} | 行为:{best['behavior_type']} | 杀死:{best['killed_count']}")
        
        # Fill remaining budget with high-kill mutants
        remaining = self.budget_k - len(selected)
        if remaining > 0:
            print(f"\n【预算填充】剩余{remaining}个名额，补充高杀死率变异体:")
            # Get all unselected mutants sorted by kill count
            unselected = [p for p in self.profiles if p['id'] not in [s['id'] for s in selected]]
            unselected.sort(key=lambda x: x['killed_count'], reverse=True)
            
            for p in unselected[:remaining]:
                selected.append(p)
                print(f"  + {p['id']} | 算子:{p['operator']} | 杀死:{p['killed_count']}")
        
        print(f"\n【最终结果】选中{len(selected)}个代表（预算k={self.budget_k}）")
        print(f"  覆盖算子类型: {len(covered_ops)}/{n_operators} - {sorted(covered_ops)}")
        
        return selected

class BaselineReducer:
    """对照组：分层KMeans（按算子分组后聚类）"""
    
    def __init__(self, profiles, budget_k):
        self.profiles = [p for p in profiles if p is not None]
        self.budget_k = budget_k
    
    def _clean_fingerprints(self, fingerprints):
        """Replace NaN/Inf with finite values"""
        # Replace NaN with 0, Inf with large finite number
        cleaned = np.nan_to_num(fingerprints, nan=0.0, posinf=1e10, neginf=-1e10)
        return cleaned
    
    def reduce(self):
        """Hierarchical clustering within operator groups"""
        # Group by operator
        op_groups = {}
        for p in self.profiles:
            op = p['operator']
            if op not in op_groups:
                op_groups[op] = []
            op_groups[op].append(p)
        
        n_ops = len(op_groups)
        print(f"算子类型数:{n_ops}, 预算k={self.budget_k}")
        
        # Allocate budget proportionally, ensure at least 1 per group if possible
        total = len(self.profiles)
        budget_per_op = {}
        
        # First pass: give 1 to each operator if budget allows
        remaining = self.budget_k
        for op in op_groups.keys():
            if remaining > 0:
                budget_per_op[op] = 1
                remaining -= 1
            else:
                budget_per_op[op] = 0
        
        # Second pass: distribute remaining proportionally
        if remaining > 0:
            for op, group in sorted(op_groups.items(), key=lambda x: -len(x[1])):
                if remaining <= 0:
                    break
                # Additional allocation based on group size
                extra = min(remaining, max(1, int(remaining * len(group) / total)))
                budget_per_op[op] += extra
                remaining -= extra
        
        selected = []
        for op, group in op_groups.items():
            k = budget_per_op.get(op, 0)
            
            # Skip if no budget allocated
            if k == 0:
                print(f"  {op}: 跳过（无预算）")
                continue
            
            # If k >= group size, select all
            if k >= len(group):
                selected.extend(group)
                print(f"  {op}: {len(group)}个变异体全选")
                continue
            
            # Ensure k is at least 1 and not larger than group
            k = max(1, min(k, len(group)))
            
            # Cluster within group
            fingerprints = np.array([p['fingerprint'] for p in group])
            
            # Clean NaN/Inf values
            fingerprints = self._clean_fingerprints(fingerprints)
            
            # Check if we have valid data
            if fingerprints.size == 0 or np.all(fingerprints == 0):
                # Fallback: select highest kill count
                exemplar = max(group, key=lambda x: x['killed_count'])
                selected.append(exemplar)
                print(f"  {op}: {len(group)}个变异体（无效指纹），选最高杀死率代表")
                continue
            
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(fingerprints, metric='euclidean')
            
            # Clean distance matrix as well
            dist_matrix = self._clean_fingerprints(dist_matrix)
            
            clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
            labels = clustering.fit_predict(dist_matrix)
            
            # Select exemplar per cluster
            for c in range(k):
                cluster_indices = [i for i, l in enumerate(labels) if l == c]
                if not cluster_indices:
                    continue
                cluster_profiles = [group[i] for i in cluster_indices]
                exemplar = max(cluster_profiles, key=lambda x: x['killed_count'])
                selected.append(exemplar)
            
            print(f"  {op}: {len(group)}个变异体分{k}簇（原预算{budget_per_op[op]}），选{k}个代表")
        
        print(f"【对照组】最终选中 {len(selected)} 个代表")
        return selected

# ==================== 5. Evaluation ====================

class Evaluator:
    def __init__(self, profiles):
        self.profiles = {p['id']: p for p in profiles if p is not None}
    
    def compute_ms(self, selected_ids):
        """Mutation Score"""
        killed = sum(1 for mid in selected_ids if self.profiles[mid]['killed_count'] > 0)
        return killed / len(selected_ids) if selected_ids else 0
    
    def compute_ftrr(self, selected_ids):
        """Fault Type Retention Rate"""
        op_types = set(self.profiles[mid]['operator'] for mid in selected_ids)
        all_ops = set(p['operator'] for p in self.profiles.values())
        return len(op_types) / len(all_ops) if all_ops else 0
    
    def evaluate(self, baseline_selected, proposed_selected):
        """Compare baseline vs proposed"""
        baseline_ms = self.compute_ms([p['id'] for p in baseline_selected])
        proposed_ms = self.compute_ms([p['id'] for p in proposed_selected])
        baseline_ftrr = self.compute_ftrr([p['id'] for p in baseline_selected])
        proposed_ftrr = self.compute_ftrr([p['id'] for p in proposed_selected])
        
        # Debug info
        print(f"  [Debug] 原始算子类型: {sorted(set(p['operator'] for p in self.profiles.values()))}")
        print(f"  [Debug] 代表算子类型: {sorted(set(p['operator'] for p in proposed_selected))}")
        print(f"  [Debug] FTRR计算: {len(set(p['operator'] for p in proposed_selected))}/{len(set(p['operator'] for p in self.profiles.values()))}={proposed_ftrr:.1%}")
        
        return {
            'baseline_ms': baseline_ms,
            'proposed_ms': proposed_ms,
            'ms_improvement': proposed_ms - baseline_ms,
            'baseline_ftrr': baseline_ftrr,
            'proposed_ftrr': proposed_ftrr,
            'ftrr_improvement': proposed_ftrr - baseline_ftrr
        }

# ==================== 6. Main Experiment ====================

def run_experiment(n_tests, budget_k, seed=42, mutants_dir="mutants"):
    """Run single experiment configuration"""
    print(f"\n  [K={budget_k}]")
    
    # Load mutants
    loader = MutantLoader(mutants_dir)
    mutant_ids = loader.get_all_ids()
    
    if len(mutant_ids) == 0:
        print(f"Error: No mutants found in {mutants_dir}")
        return None
    
    print(f"Total mutants available: {len(mutant_ids)}")
    
    # Generate test harness
    harness = TestHarness(n_tests=n_tests, seed=seed)
    
    # Profile all mutants
    profiler = BehavioralProfiler(harness, loader)
    profiles = []
    print(f"生成 {n_tests} 个 LHS 测试用例...")
    
    for i, mid in enumerate(mutant_ids, 1):
        prof = profiler.profile(mid)
        if prof:
            profiles.append(prof)
        if i % 10 == 0 or i == len(mutant_ids):
            print(f"  进度: {i}/{len(mutant_ids)} 变异体")
    
    print(f"\n所有测试完成，共处理 {len(profiles)} 个变异体")
    killed_count = sum(1 for p in profiles if p['killed_count'] > 0)
    print(f"诊断: 被杀死变异体 = {killed_count}/{len(profiles)}, Kill Matrix 形状 = ({len(profiles)}, {n_tests})")
    
    # Baseline reduction
    print(f"\n【对照组-分层KMeans】", end="")
    baseline_reducer = BaselineReducer(profiles, budget_k)
    baseline_selected = baseline_reducer.reduce()
    
    # Proposed reduction
    print(f"\n【算子优先聚类】总变异体:{len(profiles)}, 被杀死的:{killed_count}, 预算k={budget_k}")
    proposed_reducer = OperatorPriorityReducer(profiles, budget_k)
    proposed_selected = proposed_reducer.reduce()
    
    # Evaluate
    evaluator = Evaluator(profiles)
    results = evaluator.evaluate(baseline_selected, proposed_selected)
    
    print(f"\nMS: {results['baseline_ms']:.1%} → {results['proposed_ms']:.1%} ({results['ms_improvement']:+.1%}) | "
          f"FTRR: {results['baseline_ftrr']:.1%} vs {results['proposed_ftrr']:.1%}")
    
    return {
        'n_tests': n_tests,
        'budget_k': budget_k,
        **results,
        'baseline_selected': len(baseline_selected),
        'proposed_selected': len(proposed_selected)
    }

def main():
    print("=" * 70)
    print("Layer Normalization - 变异测试二维实验框架")
    print("Support M00-M120 (121 mutants)")
    print("=" * 70)
    print("======================================================================")
    print("【二维实验】测试用例规模 × 预算约束")
    print("======================================================================")
    
    test_sizes = [50]
    budgets = [5, 10, 20, 25]
    
    print(f"测试规模: {test_sizes}")
    print(f"预算取值: {budgets}")
    print("======================================================================\n")
    
    all_results = []
    
    for n_tests in test_sizes:
        print(f"======================================================================")
        print(f"【测试用例规模 N = {n_tests}】")
        print(f"======================================================================")
        
        for budget_k in budgets:
            result = run_experiment(n_tests, budget_k, seed=42, mutants_dir="mutants")
            if result:
                all_results.append(result)
        
        print()
    
    # Summary table
    print("======================================================================")
    print("【实验结果汇总】")
    print("======================================================================")
    
    # Group by test size
    from itertools import groupby
    for n_tests, group in groupby(all_results, key=lambda x: x['n_tests']):
        group_list = list(group)
        print(f"\n测试用例数 = {n_tests}:")
        print(f"{'K':<5} {'Baseline MS':<12} {'Proposed MS':<12} {'提升':<12} {'FTRR 提升':<12}")
        print("-" * 60)
        for r in group_list:
            print(f"{r['budget_k']:<5} {r['baseline_ms']:<12.1%} {r['proposed_ms']:<12.1%} "
                  f"{r['ms_improvement']:+.1%}       {r['ftrr_improvement']:+.1%}")
    
    # Save results
    csv_file = "layernorm_experiment_matrix_M120.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_tests', 'budget_k', 'baseline_ms', 'proposed_ms', 
                                               'ms_improvement', 'baseline_ftrr', 'proposed_ftrr',
                                               'ftrr_improvement', 'baseline_selected', 'proposed_selected'])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    
    print(f"\n======================================================================")
    print("【关键发现】")
    print("======================================================================")
    max_improvement = max(all_results, key=lambda x: x['ms_improvement'])
    print(f"最大 MS 提升: +{max_improvement['ms_improvement']:.1%} "
          f"(N={max_improvement['n_tests']}, K={max_improvement['budget_k']})")
    print(f"\n结果已保存到: {csv_file}")

if __name__ == "__main__":
    main()