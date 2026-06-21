"""
RMSNorm Mutant Reduction Experiment - Fixed Version
真正的行为指纹聚类 + 算子优先策略
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
    """Load mutants from generated py files (M00.py - M110.py)"""
    
    OPERATOR_MAPPING = {
        "M00": "ORIG", "M01": "AOR", "M02": "AOR", "M03": "AOR", "M04": "AOR",
        "M05": "AOR", "M06": "AOR", "M07": "AOR", "M08": "AOR", "M09": "AOR",
        "M10": "AOR", "M11": "AOR", "M12": "AOR", "M13": "ROR", "M14": "ROR",
        "M15": "ROR", "M16": "ROR", "M17": "ROR", "M18": "ROR", "M19": "ROR",
        "M20": "ROR", "M21": "CR", "M22": "CR", "M23": "CR", "M24": "CR",
        "M25": "CR", "M26": "CR", "M27": "CR", "M28": "CR", "M29": "CR",
        "M30": "CR", "M31": "SDL", "M32": "SDL", "M33": "SDL", "M34": "SDL",
        "M35": "SDL", "M36": "SDL", "M37": "VVR", "M38": "VVR", "M39": "VVR",
        "M40": "VVR", "M41": "FCR", "M42": "FCR", "M43": "FCR", "M44": "FCR",
        "M45": "FCR", "M46": "FCR", "M47": "FCR", "M48": "FCR", "M49": "FCR",
        "M50": "FCR", "M51": "ASR", "M52": "ASR", "M53": "ASR", "M54": "ASR",
        "M55": "BVR", "M56": "BVR", "M57": "BVR", "M58": "BVR", "M59": "BVR",
        "M60": "BVR", "M61": "SOR", "M62": "SOR", "M63": "SOR", "M64": "UOI",
        "M65": "UOI", "M66": "UOI", "M67": "UOI", "M68": "DTR", "M69": "DTR",
        "M70": "DTR", "M71": "EHR", "M72": "EHR", "M73": "RVR", "M74": "RVR",
        "M75": "RVR", "M76": "LVR", "M78": "CBR", "M79": "CBR", "M80": "STR",
        "M81": "STR", "M82": "IOR", "M83": "IOR", "M84": "IOR", "M85": "BOR",
        "M87": "BOR", "M90": "RMS", "M91": "RMS", "M92": "RMS", "M93": "RMS",
        "M94": "RMS", "M95": "RMS", "M96": "RMS", "M97": "RMS", "M98": "RMS",
        "M99": "RMS", "M100": "NPR", "M101": "NPR", "M102": "NPR", "M103": "MAR",
        "M104": "MAR", "M106": "DVR", "M107": "DVR", "M108": "BCR", "M109": "BCR",
        "M110": "GDR"
    }
    
    def __init__(self, mutants_dir="."):
        self.mutants_dir = mutants_dir
        self.mutant_classes = {}
        self._load_all_mutants()
    
    def _load_all_mutants(self):
        print(f"Loading mutants from {self.mutants_dir}...")
        for i in range(111):
            mid = f"M{i:02d}" if i < 100 else f"M{i}"
            filename = f"{mid}.py"
            filepath = os.path.join(self.mutants_dir, filename)
            
            if not os.path.exists(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()
                compile(source, filename, 'exec')
                
                spec = importlib.util.spec_from_file_location(f"mutant_{mid}", filepath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"mutant_{mid}"] = module
                spec.loader.exec_module(module)
                
                class_candidates = ['RMSNorm', 'LayerNorm']
                cls = None
                for name in class_candidates:
                    if hasattr(module, name):
                        cls = getattr(module, name)
                        break
                
                if cls:
                    self.mutant_classes[mid] = cls
                else:
                    print(f"  Warning: {filename} has no recognized class")
                        
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.mutant_classes)} mutants")
    
    def get_mutant(self, mid):
        return self.mutant_classes.get(mid)
    
    def get_operator_type(self, mid):
        return self.OPERATOR_MAPPING.get(mid, "UNKNOWN")
    
    def get_all_ids(self):
        return sorted(self.mutant_classes.keys())

# ==================== 2. Test Generation ====================

class TestHarness:
    def __init__(self, n_tests=50, seed=42):
        self.n_tests = n_tests
        self.seed = seed
        self.test_cases = self._generate_lhs_tests()
        self.oracle_outputs = self._compute_oracle()
        
    def _generate_lhs_tests(self):
        sampler = qmc.LatinHypercube(d=4, seed=self.seed)
        samples = sampler.random(n=self.n_tests)
        shapes = [(2, 16, 64), (4, 32, 64), (8, 64, 64), (16, 8, 64), (1, 128, 64)]
        test_cases = []
        
        np.random.seed(self.seed)
        for i, sample in enumerate(samples):
            shape_idx = int(sample[0] * len(shapes)) % len(shapes)
            batch, seq, feat = shapes[shape_idx]
            base = np.random.randn(batch, seq, feat).astype(np.float32)
            
            if sample[1] < 0.1:
                base = np.ones((batch, seq, feat)) * np.random.randn() + np.random.randn(batch, seq, feat) * 0.001
            elif sample[1] < 0.2:
                base *= 50.0
            elif sample[1] < 0.3:
                base[:, :seq//2, :] *= 0.01
                base[:, seq//2:, :] *= 100.0
            elif sample[1] < 0.4:
                base = np.ones((batch, seq, feat)) * np.random.choice([-1.0, 1.0, 2.0])
            elif sample[1] < 0.5:
                mask = np.random.rand(batch, seq, feat) > 0.8
                base = base * mask
            
            test_cases.append(torch.tensor(base))
        
        return test_cases
    
    def _compute_oracle(self):
        if not os.path.exists(os.path.join(".", "M00.py")):
            raise FileNotFoundError("M00.py not found")
        
        spec = importlib.util.spec_from_file_location("mutant_M00", "./M00.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["mutant_M00"] = module
        spec.loader.exec_module(module)
        
        SUTClass = getattr(module, 'RMSNorm', getattr(module, 'LayerNorm', None))
        if SUTClass is None:
            raise ValueError("M00.py has no RMSNorm or LayerNorm class")
        
        sut = SUTClass(64)
        sut.eval()
        with torch.no_grad():
            return [sut(x).numpy() for x in self.test_cases]
    
    def kill_mutant(self, mutant_class):
        mutant = mutant_class(64)
        mutant.eval()
        
        with torch.no_grad():
            try:
                mutant_outputs = [mutant(x).numpy() for x in self.test_cases]
                kill_vector = []
                
                for orig, mut in zip(self.oracle_outputs, mutant_outputs):
                    if not np.all(np.isfinite(mut)):
                        kill_vector.append(1)
                        continue
                    
                    diff = np.abs(orig - mut)
                    rel_diff = diff / (np.abs(orig) + 1e-8)
                    max_rel_diff = np.max(rel_diff)
                    
                    orig_rms = np.sqrt(np.mean(orig**2))
                    mut_rms = np.sqrt(np.mean(mut**2))
                    rms_deviation = abs(orig_rms - mut_rms) / (orig_rms + 1e-8)
                    
                    killed = (max_rel_diff > 1e-4) or (rms_deviation > 0.1)
                    kill_vector.append(1 if killed else 0)
                
                return np.array(kill_vector)
            except Exception:
                return np.ones(self.n_tests, dtype=int)

# ==================== 3. Behavioral Profiling ====================

class BehavioralProfiler:
    """
    增强版行为指纹提取
    """
    def __init__(self, harness, mutant_loader):
        self.harness = harness
        self.loader = mutant_loader
    
    def profile(self, mid):
        mutant_class = self.loader.get_mutant(mid)
        if mutant_class is None:
            return None
        
        kill_vector = self.harness.kill_mutant(mutant_class)
        
        mutant = mutant_class(64)
        mutant.eval()
        
        rms_values = []
        scale_factors = []
        nan_flags = []
        inf_flags = []
        output_means = []
        output_stds = []
        
        with torch.no_grad():
            for x in self.harness.test_cases:
                try:
                    output = mutant(x).numpy()
                    rms = np.sqrt(np.mean(output**2, axis=-1))
                    rms_values.append(np.mean(rms))
                    
                    x_np = x.numpy()
                    mask = np.abs(x_np) > 1e-6
                    if np.any(mask):
                        scale = np.abs(output[mask] / x_np[mask])
                        scale_factors.append(np.median(scale))
                    
                    output_means.append(np.mean(output))
                    output_stds.append(np.std(output))
                    nan_flags.append(float(np.isnan(output).any()))
                    inf_flags.append(float(np.isinf(output).any()))
                    
                except Exception:
                    nan_flags.append(1.0)
                    inf_flags.append(1.0)
                    rms_values.append(float('nan'))
                    output_means.append(0)
                    output_stds.append(0)
        
        rms_values = np.array([v for v in rms_values if np.isfinite(v)])
        
        stats = {
            'rms_mean': float(np.mean(rms_values)) if len(rms_values) > 0 else 0.0,
            'rms_std': float(np.std(rms_values)) if len(rms_values) > 0 else 0.0,
            'rms_near_one': float(np.mean(np.abs(rms_values - 1.0) < 0.1)) if len(rms_values) > 0 else 0.0,
            'output_mean': float(np.mean(output_means)),
            'output_std': float(np.mean(output_stds)),
            'nan_ratio': float(np.mean(nan_flags)),
            'inf_ratio': float(np.mean(inf_flags)),
            'scale_factor_mean': float(np.mean(scale_factors)) if scale_factors else 0.0,
        }
        
        op_type = self.loader.get_operator_type(mid)
        
        # 增强指纹：Kill Vector + 行为统计特征
        fingerprint = np.concatenate([
            kill_vector,  # 测试杀死结果（区分行为）
            [stats['rms_mean'], stats['rms_std'], stats['rms_near_one']],  # RMS特性
            [stats['nan_ratio'], stats['inf_ratio']],  # 数值稳定性
            [stats['output_mean'], stats['output_std']],  # 输出分布
            [float(np.sum(kill_vector)) / len(kill_vector)]  # 总体杀死率
        ])
        
        return {
            'id': mid,
            'operator': op_type,
            'kill_vector': kill_vector,
            'killed_count': int(np.sum(kill_vector)),
            'behavior_type': self._classify_behavior(kill_vector, stats, op_type),
            'stats': stats,
            'fingerprint': fingerprint
        }
    
    def _classify_behavior(self, kill_vector, stats, op_type):
        if stats['nan_ratio'] > 0.5:
            return "数值溢出(NaN)"
        elif stats['inf_ratio'] > 0.5:
            return "数值溢出(Inf)"
        elif np.sum(kill_vector) == 0:
            return "等价/难杀变异体"
        elif np.sum(kill_vector) < len(kill_vector) * 0.3:
            if stats['rms_near_one'] < 0.5:
                return "RMS偏离错误"
            return "边界错误"
        elif stats['rms_std'] > 1.0:
            return "不稳定RMS"
        else:
            return "常规错误"

# ==================== 4. Clustering-based Reduction ====================

class OperatorPriorityReducer:
    """
    算子优先 + 行为指纹聚类（修正版）
    策略：
    1. 按算子类型分组（Operator Priority）
    2. 在每组内使用行为指纹聚类（AgglomerativeClustering）
    3. 选择每簇中 kill_count 最高的作为代表
    """
    
    def __init__(self, profiles, budget_k):
        self.profiles = [p for p in profiles if p is not None]
        self.budget_k = budget_k
        self.operators = list(set(p['operator'] for p in self.profiles))
    
    def _clean_fingerprints(self, fingerprints):
        """清理NaN/Inf"""
        cleaned = np.nan_to_num(fingerprints, nan=0.0, posinf=1e10, neginf=-1e10)
        return cleaned
    
    def reduce(self):
        """Operator-priority + Behavioral Clustering"""
        # 1. 按算子分组
        op_groups = {}
        for p in self.profiles:
            op = p['operator']
            if op not in op_groups:
                op_groups[op] = []
            op_groups[op].append(p)
        
        n_operators = len(op_groups)
        print(f"算子类型数:{n_operators}类, 预算k={self.budget_k}")
        
        # 2. 预算分配：确保至少每类算子有预算，剩余按组内方差分配
        if self.budget_k < n_operators:
            print(f"⚠️ 预算紧张(k={self.budget_k} < 算子数{n_operators})，只覆盖{self.budget_k}类")
            # 优先选择 kill_count 高的算子类型
            op_scores = {
                op: max(p['killed_count'] for p in group)
                for op, group in op_groups.items()
            }
            selected_ops = sorted(op_scores.keys(), key=lambda x: op_scores[x], reverse=True)[:self.budget_k]
            budget_per_op = {op: 1 for op in selected_ops}
        else:
            print(f"✓ 预算充足(k={self.budget_k} >= 算子数{n_operators})，可覆盖所有算子类型")
            selected_ops = list(op_groups.keys())
            # 基础分配：每类至少1个
            budget_per_op = {op: 1 for op in selected_ops}
            remaining = self.budget_k - n_operators
            
            # 剩余预算按组内变异体数量比例分配
            total_mutants = sum(len(group) for group in op_groups.values())
            for op in selected_ops:
                if remaining <= 0:
                    break
                group_size = len(op_groups[op])
                extra = min(remaining, max(1, int(remaining * group_size / total_mutants)))
                budget_per_op[op] += extra
                remaining -= extra
        
        # 3. 在每类算子内部使用行为指纹聚类
        selected = []
        covered_ops = set()
        
        for op in selected_ops:
            group = op_groups[op]
            k = budget_per_op[op]
            
            if k >= len(group):
                # 预算充足，全选
                selected.extend(group)
                print(f"  {op}: {len(group)}个全选")
                covered_ops.add(op)
                continue
            
            # 使用行为指纹聚类
            fingerprints = np.array([p['fingerprint'] for p in group])
            fingerprints = self._clean_fingerprints(fingerprints)
            
            if len(group) <= k:
                selected.extend(group)
                covered_ops.add(op)
                continue
            
            # Agglomerative Clustering based on behavioral fingerprint
            try:
                from sklearn.metrics import pairwise_distances
                dist_matrix = pairwise_distances(fingerprints, metric='euclidean')
                dist_matrix = self._clean_fingerprints(dist_matrix)
                
                clustering = AgglomerativeClustering(
                    n_clusters=k, 
                    metric='precomputed', 
                    linkage='average'
                )
                labels = clustering.fit_predict(dist_matrix)
                
                # 选择每簇中 kill_count 最高的作为代表
                for c in range(k):
                    cluster_indices = [i for i, l in enumerate(labels) if l == c]
                    if not cluster_indices:
                        continue
                    cluster_profiles = [group[i] for i in cluster_indices]
                    exemplar = max(cluster_profiles, key=lambda x: x['killed_count'])
                    selected.append(exemplar)
                
                print(f"  {op}: {len(group)}个变异体分{k}簇（指纹聚类）")
                covered_ops.add(op)
                
            except Exception as e:
                # 聚类失败，回退到选择kill_count最高的k个
                sorted_group = sorted(group, key=lambda x: x['killed_count'], reverse=True)
                selected.extend(sorted_group[:k])
                print(f"  {op}: {len(group)}个选前{k}个（聚类失败回退）")
                covered_ops.add(op)
        
        print(f"\n【最终结果】选中{len(selected)}个代表（预算k={self.budget_k}）")
        print(f"  覆盖算子类型: {len(covered_ops)}/{n_operators}")
        
        return selected

class ElasticOperatorReducer:
    """
    弹性算子覆盖策略 (Elastic Operator Coverage)
    1. 过滤僵尸算子：只保留至少有一个变异体能杀死的算子类型
    2. 预算分配：优先覆盖高杀伤力算子类型
    3. 组内聚类：在选中的算子类型内部使用行为指纹聚类
    """
    
    def __init__(self, profiles, budget_k, min_kill_threshold=1):
        """
        Args:
            profiles: 变异体画像列表
            budget_k: 预算约束
            min_kill_threshold: 算子有效性阈值，默认1（至少杀死1个测试用例才算有效）
        """
        self.profiles = [p for p in profiles if p is not None]
        self.budget_k = budget_k
        self.min_kill_threshold = min_kill_threshold
        self.viable_ops = self._identify_viable_operators()
    
    def _identify_viable_operators(self):
        """识别有效算子：该类型下最大杀死数 >= 阈值的算子"""
        op_max_kills = {}
        for p in self.profiles:
            op = p['operator']
            if op not in op_max_kills:
                op_max_kills[op] = 0
            op_max_kills[op] = max(op_max_kills[op], p['killed_count'])
        
        viable = {op for op, max_kill in op_max_kills.items() 
                 if max_kill >= self.min_kill_threshold}
        
        print(f"【弹性覆盖】有效算子: {len(viable)}/{len(op_max_kills)} 类")
        print(f"  过滤掉的算子: {[op for op in op_max_kills if op not in viable]}")
        return viable
    
    def _clean_fingerprints(self, fingerprints):
        """清理NaN/Inf值"""
        return np.nan_to_num(fingerprints, nan=0.0, posinf=1e10, neginf=-1e10)
    
    def reduce(self):
        #僵尸算子去除
        ZOMBIE_OPS = {'ORIG', 'MAR', 'DVR', 'GDR', 'DTR', 'LVR', 'CBR', 'STR', 'BOR'}
        viable_profiles = [p for p in self.profiles if p['operator'] not in ZOMBIE_OPS]
        """执行弹性约简"""
        # 1. 按算子分组，只保留有效算子
        op_groups = {}
        for p in viable_profiles:  #用去掉僵尸算子后的算子集
            op = p['operator']
            if op in self.viable_ops:  # 过滤僵尸算子
                if op not in op_groups:
                    op_groups[op] = []
                op_groups[op].append(p)
        
        if not op_groups:
            print("警告：没有有效算子，回退到全局Top-K")
            sorted_profiles = sorted(self.profiles, 
                                   key=lambda x: x['killed_count'], 
                                   reverse=True)
            return sorted_profiles[:self.budget_k]
        
        # 2. 计算每个有效算子的代表杀伤力（用于排序）
        op_effectiveness = {
            op: max(p['killed_count'] for p in group)
            for op, group in op_groups.items()
        }
        
        n_viable = len(op_groups)
        print(f"预算k={self.budget_k}, 有效算子数={n_viable}")
        
        # 3. 选择要覆盖的算子类型（预算不足时选Top-K有效算子）
        if self.budget_k >= n_viable:
            print(f"✓ 预算充足，覆盖所有{n_viable}个有效算子类型")
            selected_ops = list(op_groups.keys())
            budget_per_op = {op: 1 for op in selected_ops}
            remaining = self.budget_k - n_viable
            
            # 剩余预算按组内方差/大小分配
            total_size = sum(len(group) for group in op_groups.values())
            for op in selected_ops:
                if remaining <= 0:
                    break
                extra = min(remaining, 
                           max(1, int(remaining * len(op_groups[op]) / total_size)))
                budget_per_op[op] += extra
                remaining -= extra
        else:
            print(f"⚠️ 预算紧张，选择Top-{self.budget_k}有效算子")
            sorted_ops = sorted(op_effectiveness.keys(), 
                              key=lambda x: op_effectiveness[x], 
                              reverse=True)
            selected_ops = sorted_ops[:self.budget_k]
            budget_per_op = {op: 1 for op in selected_ops}
        
        # 4. 在选中的算子类型内部进行行为指纹聚类
        selected = []
        covered_ops = set()
        
        for op in selected_ops:
            group = op_groups[op]
            k = budget_per_op[op]
            
            # 如果组内数量 <= 预算，全选
            if len(group) <= k:
                selected.extend(group)
                covered_ops.add(op)
                print(f"  {op}: {len(group)}个全选 (kill_max={op_effectiveness[op]})")
                continue
            
            # 行为指纹聚类
            fingerprints = np.array([p['fingerprint'] for p in group])
            fingerprints = self._clean_fingerprints(fingerprints)
            
            try:
                from sklearn.metrics import pairwise_distances
                dist_matrix = pairwise_distances(fingerprints, metric='euclidean')
                dist_matrix = self._clean_fingerprints(dist_matrix)
                
                clustering = AgglomerativeClustering(
                    n_clusters=k, 
                    metric='precomputed', 
                    linkage='average'
                )
                labels = clustering.fit_predict(dist_matrix)
                
                # 选择每簇中kill_count最高的作为代表
                for c in range(k):
                    cluster_indices = [i for i, l in enumerate(labels) if l == c]
                    if not cluster_indices:
                        continue
                    cluster_profiles = [group[i] for i in cluster_indices]
                    exemplar = max(cluster_profiles, key=lambda x: x['killed_count'])
                    selected.append(exemplar)
                
                print(f"  {op}: {len(group)}个分{k}簇（行为聚类）")
                covered_ops.add(op)
                
            except Exception as e:
                # 聚类失败，回退到Top-K
                sorted_group = sorted(group, key=lambda x: x['killed_count'], reverse=True)
                selected.extend(sorted_group[:k])
                print(f"  {op}: 聚类失败，选Top-{k} (回退)")
                covered_ops.add(op)
        
        print(f"\n【弹性覆盖结果】选中{len(selected)}个代表")
        print(f"  覆盖有效算子: {len(covered_ops)}/{n_viable}")
        print(f"  实际FTRR(有效): {len(covered_ops)/n_viable:.1%}")
        
        return selected


class BaselineReducer:
    """
    对照组：纯行为指纹聚类（不考虑算子类型）
    """
    
    def __init__(self, profiles, budget_k):
        self.profiles = [p for p in profiles if p is not None]
        self.budget_k = budget_k
    
    def _clean_fingerprints(self, fingerprints):
        cleaned = np.nan_to_num(fingerprints, nan=0.0, posinf=1e10, neginf=-1e10)
        return cleaned
    
    def reduce(self):
        """Pure Behavioral Clustering without operator priority"""
        if self.budget_k >= len(self.profiles):
            return self.profiles
        
        fingerprints = np.array([p['fingerprint'] for p in self.profiles])
        fingerprints = self._clean_fingerprints(fingerprints)
        
        try:
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(fingerprints, metric='euclidean')
            dist_matrix = self._clean_fingerprints(dist_matrix)
            
            clustering = AgglomerativeClustering(
                n_clusters=self.budget_k,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(dist_matrix)
            
            selected = []
            for c in range(self.budget_k):
                cluster_indices = [i for i, l in enumerate(labels) if l == c]
                if not cluster_indices:
                    continue
                cluster_profiles = [self.profiles[i] for i in cluster_indices]
                exemplar = max(cluster_profiles, key=lambda x: x['killed_count'])
                selected.append(exemplar)
            
            print(f"【对照组-纯聚类】{len(self.profiles)}个变异体分{self.budget_k}簇，选{len(selected)}个代表")
            return selected
            
        except Exception as e:
            # 回退到随机选择
            import random
            random.seed(42)
            selected = random.sample(self.profiles, self.budget_k)
            print(f"【对照组】聚类失败，随机选择{self.budget_k}个")
            return selected

# ==================== 5. Evaluation ====================

class Evaluator:
    def __init__(self, profiles):
        self.profiles = {p['id']: p for p in profiles if p is not None}
    
    def compute_ms(self, selected_ids):
        killed = sum(1 for mid in selected_ids if self.profiles[mid]['killed_count'] > 0)
        return killed / len(selected_ids) if selected_ids else 0
    
    def compute_ftrr(self, selected_ids):
        op_types = set(self.profiles[mid]['operator'] for mid in selected_ids)
        all_ops = set(p['operator'] for p in self.profiles.values())
        return len(op_types) / len(all_ops) if all_ops else 0
    
    def compute_behavior_coverage(self, selected_ids):
        """计算行为类型覆盖率（新增指标）"""
        selected_behaviors = set(self.profiles[mid]['behavior_type'] for mid in selected_ids)
        all_behaviors = set(p['behavior_type'] for p in self.profiles.values())
        return len(selected_behaviors) / len(all_behaviors) if all_behaviors else 0
    
    def evaluate(self, baseline_selected, proposed_selected):
        baseline_ids = [p['id'] for p in baseline_selected]
        proposed_ids = [p['id'] for p in proposed_selected]
        
        baseline_ms = self.compute_ms(baseline_ids)
        proposed_ms = self.compute_ms(proposed_ids)
        baseline_ftrr = self.compute_ftrr(baseline_ids)
        proposed_ftrr = self.compute_ftrr(proposed_ids)
        baseline_bcr = self.compute_behavior_coverage(baseline_ids)  # Behavior Coverage Rate
        proposed_bcr = self.compute_behavior_coverage(proposed_ids)
        
        return {
            'baseline_ms': baseline_ms,
            'proposed_ms': proposed_ms,
            'ms_improvement': proposed_ms - baseline_ms,
            'baseline_ftrr': baseline_ftrr,
            'proposed_ftrr': proposed_ftrr,
            'ftrr_improvement': proposed_ftrr - baseline_ftrr,
            'baseline_bcr': baseline_bcr,
            'proposed_bcr': proposed_bcr,
            'bcr_improvement': proposed_bcr - baseline_bcr
        }

# ==================== 6. Main Experiment ====================

def run_experiment(n_tests, budget_k, seed=42, mutants_dir="."):
    print(f"\n  [N={n_tests}, K={budget_k}]")
    
    loader = MutantLoader(mutants_dir)
    mutant_ids = loader.get_all_ids()
    
    if len(mutant_ids) == 0:
        print(f"Error: No mutants found")
        return None
    
    print(f"Total mutants available: {len(mutant_ids)}")
    
    harness = TestHarness(n_tests=n_tests, seed=seed)
    profiler = BehavioralProfiler(harness, loader)
    profiles = []
    
    print(f"生成 {n_tests} 个 LHS 测试用例...")
    for i, mid in enumerate(mutant_ids, 1):
        prof = profiler.profile(mid)
        if prof:
            profiles.append(prof)
        if i % 20 == 0 or i == len(mutant_ids):
            print(f"  进度: {i}/{len(mutant_ids)}")
    
    print(f"\n所有测试完成，共处理 {len(profiles)} 个变异体")
    killed_count = sum(1 for p in profiles if p['killed_count'] > 0)
    print(f"诊断: 被杀死变异体 = {killed_count}/{len(profiles)}")
    
    # 显示行为分布
    behavior_dist = {}
    for p in profiles:
        bt = p['behavior_type']
        behavior_dist[bt] = behavior_dist.get(bt, 0) + 1
    print("行为分布:", dict(sorted(behavior_dist.items(), key=lambda x: -x[1])[:5]))
    
    # 对照组：纯行为聚类（不考虑算子）
    print(f"\n【对照组-纯行为聚类】", end="")
    baseline_reducer = BaselineReducer(profiles, budget_k)
    baseline_selected = baseline_reducer.reduce()
    
    # 实验组：算子优先 + 行为指纹聚类
    print(f"\n【算子优先+行为聚类】预算k={budget_k}")
    proposed_reducer = ElasticOperatorReducer(profiles, budget_k, min_kill_threshold=1)
    proposed_selected = proposed_reducer.reduce()
    
    # 评估
    evaluator = Evaluator(profiles)
    results = evaluator.evaluate(baseline_selected, proposed_selected)
    
    print(f"\nMS: {results['baseline_ms']:.1%} → {results['proposed_ms']:.1%} ({results['ms_improvement']:+.1%}) | "
          f"FTRR: {results['baseline_ftrr']:.1%} vs {results['proposed_ftrr']:.1%} | "
          f"BCR: {results['baseline_bcr']:.1%} vs {results['proposed_bcr']:.1%}")
    
    return {
        'n_tests': n_tests,
        'budget_k': budget_k,
        **results,
        'baseline_selected': len(baseline_selected),
        'proposed_selected': len(proposed_selected)
    }

def main():
    print("=" * 70)
    print("RMSNorm - 算子优先 vs 纯行为聚类 对比实验")
    print("=" * 70)
    
    test_sizes = [50, 100]
    budgets = [5, 10, 20, 30, 50]
    
    print(f"测试规模: {test_sizes}")
    print(f"预算取值: {budgets}")
    
    all_results = []
    
    for n_tests in test_sizes:
        print(f"\n{'='*70}")
        print(f"【测试用例规模 N = {n_tests}】")
        print(f"{'='*70}")
        
        for budget_k in budgets:
            result = run_experiment(n_tests, budget_k, seed=42, mutants_dir=".")
            if result:
                all_results.append(result)
    
    # 汇总
    print(f"\n{'='*70}")
    print("【实验结果汇总】")
    print(f"{'='*70}")
    
    from itertools import groupby
    for n_tests, group in groupby(all_results, key=lambda x: x['n_tests']):
        group_list = list(group)
        print(f"\n测试用例数 = {n_tests}:")
        print(f"{'K':<5} {'Baseline MS':<12} {'Proposed MS':<12} {'提升':<10} {'BCR提升':<10}")
        print("-" * 60)
        for r in group_list:
            print(f"{r['budget_k']:<5} {r['baseline_ms']:<12.1%} {r['proposed_ms']:<12.1%} "
                  f"{r['ms_improvement']:+.1%}     {r['bcr_improvement']:+.1%}")
    
    # 保存
    csv_file = "rmsnorm_behavior_clustering.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_tests', 'budget_k', 'baseline_ms', 'proposed_ms', 
                                               'ms_improvement', 'baseline_ftrr', 'proposed_ftrr',
                                               'ftrr_improvement', 'baseline_bcr', 'proposed_bcr',
                                               'bcr_improvement', 'baseline_selected', 'proposed_selected'])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    
    print(f"\n结果已保存到: {csv_file}")

if __name__ == "__main__":
    main()