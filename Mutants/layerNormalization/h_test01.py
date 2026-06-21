import torch
import warnings
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import random
import importlib.util
import os
from typing import Dict, List, Callable, Optional, Tuple, Any
from Mutants.mutant_ananlysis import detect_fingerprint_collisions, analyze_single_operator,print_operator_summary

from Mutants.experiment_1 import plot_coverage,plot_cm_both,plot_km_fp_confusion_heatmap,plot_km_layer_heatmap

from Mutants.experiment_rq2 import compute_rq2_metrics,plot_fingerprint_tsne,extract_case_studies,plot_cases
from Mutants.experiment_rq3 import run_rq3_experiment
from Mutants.experiment_rq4 import run_rq4_experiment_a,run_rq4_experiment_b

all_behavior_types = [
    # 基础数值异常 (0-2)
    "invalid_output",      # 0: 非法输出/None
    "nan",                 # 1: 包含NaN
    "inf",                 # 2: 包含Inf
    
    # 统计特性异常 (3-6)
    "mean_nonzero",        # 3: 均值不为0(标准化失败)
    "variance_nonone",     # 4: 方差不为1(标准化失败)
    "mean_variance_mismatch",  # 5: 均值≠0且方差≠1(完全未标准化)
    "zero_variance",       # 6: 方差为0(无法标准化)
    
    # 仿射变换异常 (7-9)
    "weight_scale_error",      # 7: weight缩放错误(输出范围异常)
    "bias_shift_error",        # 8: bias偏移错误(输出中心偏移)
    "affine_transformation_loss",  # 9: 仿射变换丢失(weight/bias未应用)
    
    # 数值稳定性异常 (10-13)
    "eps_ineffective",     # 10: eps无效(仍出现除零/数值不稳定)
    "catastrophic_cancellation",  # 11: 灾难性抵消(均值/方差计算精度丢失)
    "overflow_risk",       # 12: 溢出风险(输入过大)
    "underflow_risk",      # 13: 下溢风险(输入过小)
    
    # 维度/形状异常 (14-17)
    "dimension_mismatch",      # 14: 维度不匹配(normalized_shape错误)
    "wrong_normalization_axis",  # 15: 错误归一化轴(非最后一维)
    "keepdim_error",           # 16: keepdim语义错误
    "broadcasting_error",      # 17: 广播错误(weight/bias形状不匹配)
    
    # 语义/逻辑异常 (18-21)
    "distribution_distortion",  # 18: 分布扭曲(相对顺序改变)
    "scale_invariance_violation",  # 19: 尺度不变性违反(输入缩放后输出变化)
    "shift_invariance_violation",  # 20: 平移不变性违反(输入平移后输出变化)
    "gradient_flow_blocked"   # 21: 梯度流阻断(梯度消失/爆炸)   

]

# ============================================
# 1.测试数据生成
# ============================================
def generate_lhs_samples(
    n: int, 
    seed: Optional[int] = None, 
    normalized_shape: int = 64,
    batch_size: int = 32,
    extra_dims: int = 0  # 额外维度，如序列长度等
) -> List[Dict[str, Any]]:
    """
    生成Layer Normalization的测试用例
    
    Args:
        n: 需要生成的测试用例总数
        seed: 随机种子
        normalized_shape: 归一化的特征维度
        batch_size: 批次大小
        extra_dims: 额外的维度数量（如序列长度），0表示2D输入[batch, features]
    
    Returns:
        测试用例列表，每个测试用例包含:
        - 'input': 输入张量
        - 'strategy': 生成策略 ('normal', 'targeted', 'boundary')
        - 'target_type': 目标异常类型（仅针对策略2）
        - 'description': 测试用例描述
    """
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # 计算各策略数量
    n_normal = int(n * 0.4)
    n_targeted = int(n * 0.4)
    n_boundary = n - n_normal - n_targeted
    
    test_cases = []
    
    # ==================== 策略1: 常规功能测试 (40%) ====================
    def generate_normal_test_cases(count: int) -> List[Dict[str, Any]]:
        """生成常规功能测试用例"""
        cases = []
        
        # 定义输入分布策略
        distribution_modes = [
            'normal',      # 标准正态分布
            'uniform',     # 均匀分布
            'mixed_range', # 混合范围（含极端值）
            'near_zero',   # 接近零
            'large_scale'  # 大尺度值
        ]
        
        for i in range(count):
            # 随机选择分布模式
            mode = np.random.choice(distribution_modes)
            
            # 构建输入形状
            shape = [batch_size]
            if extra_dims > 0:
                shape.extend([np.random.randint(1, 10) for _ in range(extra_dims)])
            shape.append(normalized_shape)
            
            # 根据模式生成数据
            if mode == 'normal':
                x = torch.randn(*shape)
                desc = f"标准正态分布"
                
            elif mode == 'uniform':
                x = torch.rand(*shape) * 2 - 1  # [-1, 1]均匀分布
                desc = f"均匀分布[-1,1]"
                
            elif mode == 'mixed_range':
                # 混合正常值和极端值（10%极端值）
                x = torch.randn(*shape)
                extreme_mask = torch.rand(*shape) < 0.1
                x[extreme_mask] = torch.randn(extreme_mask.sum()) * 10  # 极端值
                desc = f"混合分布(90%正常,10%极端值)"
                
            elif mode == 'near_zero':
                x = torch.randn(*shape) * 1e-6
                desc = f"接近零分布(1e-6)"
                
            else:  # large_scale
                x = torch.randn(*shape) * 100
                desc = f"大尺度分布(标准差100)"
            
            cases.append({
                'input': x,
                'strategy': 'normal',
                'target_type': None,
                'description': f"常规测试-{desc}"
            })
        
        return cases
    
    # ==================== 策略2: 针对性故障触发 (40%) ====================
    def generate_targeted_test_cases(count: int) -> List[Dict[str, Any]]:
        """生成针对性故障触发测试用例"""
        cases = []
        
        # 定义每个异常类型的生成器
        type_generators = {
            # 基础数值异常
            "invalid_output": lambda: {
                'input': None,  # 特殊标记
                'special': 'none_input'
            },
            "nan": lambda: {
                'input': torch.tensor([[[float('nan')]] * normalized_shape]),
                'special': 'nan'
            },
            "inf": lambda: {
                'input': torch.tensor([[[float('inf')]] * normalized_shape]),
                'special': 'inf'
            },
            
            # 统计特性异常
            "mean_nonzero": lambda: {
                'input': torch.randn(batch_size, normalized_shape) + 10,  # 偏移均值
                'expected_issue': 'mean_not_zero'
            },
            "variance_nonone": lambda: {
                'input': torch.randn(batch_size, normalized_shape) * 5,  # 改变方差
                'expected_issue': 'variance_not_one'
            },
            "zero_variance": lambda: {
                'input': torch.ones(batch_size, normalized_shape),  # 常数输入
                'expected_issue': 'zero_variance'
            },
            # "negative_variance": lambda: {
            #     'input': torch.randn(batch_size, normalized_shape),
            #     'special': 'negative_variance_trigger',
            #     'description': '通过数值不稳定的计算触发负方差',
            #     # 注意：这需要变异体本身产生负方差，测试用例只能提供输入
            # },
            
            # 仿射变换异常
            # "weight_scale_error": lambda: {
            #     'input': torch.randn(batch_size, normalized_shape),
            #     'special': 'extreme_weight'
            # },
            "weight_scale_error": lambda: {
                'input': torch.randn(batch_size, normalized_shape) * 1000,  # 大尺度输入
                'elementwise_affine': True,
                'special': 'extreme_input_for_weight',
                'description': '大尺度输入，若weight异常会导致输出范围错误'
            },
            "bias_shift_error": lambda: {
                'input': torch.randn(batch_size, normalized_shape),
                'special': 'extreme_bias'
            },
            # "elementwise_affine_misconfig": lambda: {
            #     'input': torch.randn(batch_size, normalized_shape),
            #     'elementwise_affine': True,
            #     'special': 'check_affine_config',
            #     'description': '测试仿射变换配置是否正确应用'
            # },
            # "elementwise_affine_misconfig_false": lambda: {
            #     'input': torch.randn(batch_size, normalized_shape),
            #     'elementwise_affine': False,
            #     'special': 'check_affine_config_false',
            #     'description': '测试仿射变换禁用时是否正确'
            # },
            
            # 数值稳定性异常
            # "eps_ineffective": lambda: {
            #     'input': torch.zeros(batch_size, normalized_shape),  # 零方差需要eps
            #     'expected_issue': 'eps_required'
            # },
            "eps_ineffective": lambda: {
                'input': torch.ones(batch_size, normalized_shape),  # 方差为0的输入
                'eps': 1e-20,  # 极小的eps
                'description': '零方差输入 + 极小eps，应触发eps无效'
            },
            "eps_ineffective_v2": lambda: {
                'input': torch.zeros(batch_size, normalized_shape),  # 全零输入
                'eps': 1e-30,  # 极端小的eps
                'description': '零输入 + 极小eps'
            },
            "overflow_risk": lambda: {
                'input': torch.randn(batch_size, normalized_shape) * 1e8,
                'expected_issue': 'overflow_risk'
            },
            "underflow_risk": lambda: {
                'input': torch.randn(batch_size, normalized_shape) * 1e-8,
                'expected_issue': 'underflow_risk'
            },
            
            # 维度/形状异常
            "dimension_mismatch": lambda: {
                'input': torch.randn(batch_size, normalized_shape + 10),  # 维度不匹配
                'special': 'dimension_mismatch'
            },
            "wrong_normalization_axis": lambda: {
                'input': torch.randn(batch_size, normalized_shape),
                'special': 'wrong_axis'
            },
            "wrong_normalization_axis": lambda: {
                'input': torch.randn(batch_size, 10, normalized_shape),  # 3D输入 [batch, seq, features]
                'normalized_shape': normalized_shape,
                'special': 'wrong_axis_test',
                'description': '3D输入，变异体可能错误地对batch或seq维度归一化'
            },
            "broadcasting_error": lambda: {
                'input': torch.randn(batch_size, normalized_shape),
                'special': 'broadcast_error'
            },
            "dimension_mismatch": lambda: {
                'input': torch.randn(batch_size, normalized_shape + 10),  # 故意维度不匹配
                'normalized_shape': normalized_shape,  # 但指定了不同的 normalized_shape
                'description': '输入维度与 normalized_shape 不匹配'
            },
            "dimension_mismatch_2d_3d": lambda: {
                'input': torch.randn(batch_size, 10, normalized_shape),  # 3D输入
                'normalized_shape': normalized_shape,  # 但预期是2D
                'description': '输入维度数错误'
            },
            
            # 语义/逻辑异常
            "distribution_distortion": lambda: {
                'input': torch.randn(batch_size, normalized_shape) * torch.randn(normalized_shape),  # 非各向同性
                'expected_issue': 'distribution_distortion'
            },
            "scale_invariance_violation": lambda: {
                'input': torch.randn(batch_size, normalized_shape),
                'special': 'scale_test'
            },
            "gradient_flow_blocked": lambda: {
                'input': torch.randn(batch_size, normalized_shape),
                'requires_grad': True,
                'expected_issue': 'gradient_flow'
            },

            # ===== 补充：zero_variance 相关的增强 =====
            "zero_variance_constant": lambda: {
                'input': torch.full((batch_size, normalized_shape), 5.0),  # 常数输入
                'description': '常数输入，方差为0'
            },
            "zero_variance_single_sample": lambda: {
                'input': torch.randn(1, normalized_shape),  # 单样本
                'description': '单样本输入，可能触发方差计算问题'
            },
            

        }
        
        # 为每个类型至少生成一个用例
        all_types = list(type_generators.keys())
        for target_type in all_types:
            if len(cases) >= count:
                break
            try:
                test_case = type_generators[target_type]()
                # 构建完整的测试用例
                case = {
                    'input': test_case.get('input'),
                    'strategy': 'targeted',
                    'target_type': target_type,
                    'description': f"针对性测试-{target_type}",
                    'special': test_case.get('special'),
                    'requires_grad': test_case.get('requires_grad', False)
                }
                
                # 处理特殊输入
                if case['input'] is None and 'special' in case:
                    # 特殊输入将在测试时动态创建
                    pass
                elif isinstance(case['input'], torch.Tensor):
                    # 确保形状正确
                    pass
                else:
                    # 默认生成正常输入
                    shape = [batch_size, normalized_shape]
                    case['input'] = torch.randn(*shape)
                
                cases.append(case)
            except Exception as e:
                print(f"生成{target_type}测试用例失败: {e}")
                continue
        
        # 如果数量不够，通过扰动现有用例补充
        while len(cases) < count and cases:
            # 随机选择一个已有用例进行扰动
            base_case = random.choice(cases)
            if isinstance(base_case['input'], torch.Tensor):
                # 添加微小扰动
                perturbed_input = base_case['input'].clone()
                noise = torch.randn_like(perturbed_input) * 0.01
                perturbed_input = perturbed_input + noise
                
                new_case = {
                    'input': perturbed_input,
                    'strategy': 'targeted',
                    'target_type': base_case['target_type'],
                    'description': f"{base_case['description']}(扰动副本)",
                    'special': base_case.get('special'),
                    'requires_grad': base_case.get('requires_grad', False)
                }
                cases.append(new_case)
        
        return cases[:count]
    
    # ==================== 策略3: 边界攻击 (20%) ====================
    def generate_boundary_test_cases(count: int) -> List[Dict[str, Any]]:
        """生成边界攻击测试用例"""
        cases = []
        
        attack_modes = [
            'overflow',      # 上溢攻击
            'underflow',     # 下溢攻击
            'precision',     # 精度极限攻击
            'mixed_scale',   # 混合尺度攻击
            'extreme_ratio', # 极端比例攻击
            'near_epsilon'   # 接近epsilon攻击
        ]
        
        for i in range(count):
            mode = np.random.choice(attack_modes)
            
            shape = [batch_size, normalized_shape]
            
            if mode == 'overflow':
                # 极大值攻击
                x = torch.randn(*shape) * 1e10
                desc = "上溢攻击-极大值"
                
            elif mode == 'underflow':
                # 极小值攻击
                x = torch.randn(*shape) * 1e-10
                desc = "下溢攻击-极小值"
                
            elif mode == 'precision':
                # 精度极限攻击
                x = torch.randn(*shape)
                # 添加微小扰动到精度极限
                x = x + torch.randn(*shape) * 1e-7
                desc = "精度极限攻击"
                
            elif mode == 'mixed_scale':
                # 混合尺度攻击
                x = torch.randn(*shape)
                # 部分维度极大，部分极小
                split = normalized_shape // 2
                x[:, :split] *= 1e8
                x[:, split:] *= 1e-8
                desc = "混合尺度攻击"
                
            elif mode == 'extreme_ratio':
                # 极端比例攻击
                x = torch.randn(*shape)
                # 创建极端差异
                x[:, 0] *= 1e6  # 第一个特征极大
                x[:, 1] *= 1e-6  # 第二个特征极小
                desc = "极端比例攻击"
                
            else:  # near_epsilon
                # 接近epsilon的攻击
                eps = 1e-5
                x = torch.randn(*shape)
                # 制造方差接近eps的情况
                x = x * torch.sqrt(torch.tensor(eps)) + torch.randn(*shape) * 1e-3
                desc = "接近epsilon攻击"
            
            cases.append({
                'input': x,
                'strategy': 'boundary',
                'target_type': mode,
                'description': f"边界攻击-{desc}",
                'attack_mode': mode
            })
        
        return cases
    
    # 生成各策略测试用例
    print(f"生成测试用例: 总数={n}, 常规={n_normal}, 针对性={n_targeted}, 边界={n_boundary}")
    
    test_cases.extend(generate_normal_test_cases(n_normal))
    test_cases.extend(generate_targeted_test_cases(n_targeted))
    test_cases.extend(generate_boundary_test_cases(n_boundary))
    
    # 如果超过n个，随机删除一些
    if len(test_cases) > n:
        test_cases = random.sample(test_cases, n)
    
    # 打乱顺序
    random.shuffle(test_cases)
    
    return test_cases

# ==========================
# 2.载入 Oracle & Mutants
# ==========================
def load_oracle() -> Optional[Callable]:
    """
    加载原始代码 M00.py 中的 LayerNorm 类
    
    Returns:
        LayerNorm 类对象，加载失败返回 None
    """
    try:
        spec = importlib.util.spec_from_file_location('M00', 'mutants/M00.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 返回 LayerNorm 类（原始代码中定义的类名）
        return mod.LayerNorm
    except Exception as e:
        print(f"[WARN] Failed to load Oracle (M00.py): {e}")
        return None

def load_mutants(mutant_files: list) -> Dict[str, Callable]:
    """
    加载变异体文件 M01.py 到 M120.py 中的 LayerNorm 类
    
    Args:
        mutant_files: 变异体文件路径列表，如 ['M01.py', 'M02.py', ...]
    
    Returns:
        字典，key为变异体名称（如'M01'），value为LayerNorm类对象
    """
    mutant_funcs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mf in mutant_files:
            # 提取文件名（不含扩展名）作为变异体标识
            name = os.path.splitext(os.path.basename(mf))[0]
            
            try:
                spec = importlib.util.spec_from_file_location(name, mf)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                
                # 获取变异体中的 LayerNorm 类
                if hasattr(mod, 'LayerNorm'):
                    mutant_funcs[name] = mod.LayerNorm
                else:
                    print(f"[WARN] {mf} does not have LayerNorm class, skipping...")
                    
            except Exception as e:
                print(f"[WARN] Failed to load {mf}: {e}")
    
    return mutant_funcs


# ==========================
# 3.三层决策引擎
# ==========================

def detect_violations(S: Any, tol: float = 1e-7, x: Optional[torch.Tensor] = None,normalized_shape: Optional[int] = None) -> List[str]:
    """
    检测LayerNorm输出的数学性质违规。
    
    Args:
        S: LayerNorm输出张量（torch.Tensor或numpy数组）
        tol: 数值容差
        x: 原始输入张量（可选，用于检测涉及输入输出的违规）
    
    Returns:
        检测到的违规类型列表
    """
    violations = []

    # 基础数值异常
    if S is None:
        return ["invalid_output"]
    if not isinstance(S, (torch.Tensor, np.ndarray)):
        return ["invalid_output"]
    if isinstance(S, torch.Tensor):
        S_np = S.detach().cpu().numpy()
    else:
        S_np = S

    if S_np.size == 0:
        return ["invalid_output"]
    
    # 使用 errstate 忽略 NaN/Inf 相关的警告
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        if np.isnan(S_np).any():
            violations.append("nan")
        if np.isinf(S_np).any():
            violations.append("inf")

        # 统计特性异常（标准化后应均值≈0，方差≈1）
        axis = -1
        mean = np.mean(S_np, axis=axis, keepdims=True)
        var = np.var(S_np, axis=axis, keepdims=True)

        if not np.allclose(mean, 0, atol=tol):
            violations.append("mean_nonzero")
        if not np.allclose(var, 1, atol=tol):
            violations.append("variance_nonone")
        if not np.allclose(mean, 0, atol=tol) and not np.allclose(var, 1, atol=tol):
            violations.append("mean_variance_mismatch")
        # if np.any(var < 0):
        #     violations.append("negative_variance")
        # 检测负方差（直接）
        # if np.any(var < 0):
        #     violations.append("negative_variance")        
        # # 检测数值不稳定导致的极小负值（容差内）
        # if np.any(var < -tol):
        #     violations.append("negative_variance")
        # if np.any(var < tol):
        #     violations.append("zero_variance")

        # 仿射变换异常
        if np.max(np.abs(S_np)) > 1e6:
            violations.append("weight_scale_error")
        if np.mean(S_np) > 1e3 or np.mean(S_np) < -1e3:
            violations.append("bias_shift_error")
        
        if x is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            
            # 确保形状相同才能比较
            if S_np.shape == x_np.shape:
                if np.allclose(S_np, x_np, atol=tol):
                    violations.append("affine_transformation_loss")

        # 数值稳定性异常
        eps = 1e-5
        if np.any(var < eps) and not np.any(np.isinf(S_np)):
            violations.append("eps_ineffective")
        
        if x is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            if np.max(np.abs(x_np)) > 1e5 and np.max(np.abs(S_np)) < 1e-5:
                violations.append("catastrophic_cancellation")
        
        if np.any(np.abs(S_np) > 1e10):
            violations.append("overflow_risk")
        if np.any(np.abs(S_np) < 1e-10) and np.any(np.abs(S_np) > 0):
            violations.append("underflow_risk")

        # 维度/形状异常
        if S_np.ndim == 0:
            violations.append("dimension_mismatch")
        
        if x is not None and x.ndim > 1:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            
            # 检查形状是否匹配
            if S_np.shape != x_np.shape:
                violations.append("keepdim_error")
            else:
                # 只当形状相同时才检查归一化轴
                var_other = np.var(S_np, axis=tuple(range(x_np.ndim-1)), keepdims=True)
                if np.all(var_other < tol):
                    violations.append("wrong_normalization_axis")
        
        # 广播错误：输出各样本完全相同（批次内）
        if S_np.ndim >= 2 and S_np.shape[0] > 1:
            if np.allclose(S_np, S_np[0:1], atol=tol):
                violations.append("broadcasting_error")

    # 语义/逻辑异常（这些计算也可能产生警告，需要单独包裹）
    if x is not None:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # 确保形状相同才能进行后续检查
        if S_np.shape == x_np.shape:
            # 分布扭曲：比较排序一致性（沿最后一维）
            if x_np.shape[-1] == S_np.shape[-1]:
                try:
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                        x_order = np.argsort(x_np, axis=-1)
                        s_order = np.argsort(S_np, axis=-1)
                        if not np.allclose(x_order, s_order, atol=tol):
                            violations.append("distribution_distortion")
                except Exception:
                    pass  # 排序比较失败时跳过

    # 尺度不变性和平移不变性检查（需要逐样本处理）
    if x is not None and S_np.shape == x_np.shape:
        try:
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                # 沿最后一维计算统计量
                scale = np.std(x_np, axis=-1, keepdims=True)
                out_scale = np.std(S_np, axis=-1, keepdims=True)
                
                # 只当没有nan/inf且数据有效时才计算相关系数
                if (np.isfinite(scale).all() and np.isfinite(out_scale).all() and 
                    np.any(scale > 1e-6) and np.any(out_scale > 1e-6)):
                    # 展平但保持样本对应关系
                    scale_flat = scale.flatten()
                    out_scale_flat = out_scale.flatten()
                    if len(scale_flat) == len(out_scale_flat) and len(scale_flat) > 1:
                        corr = np.corrcoef(scale_flat, out_scale_flat)[0, 1]
                        if not np.isnan(corr) and corr > 0.5:
                            violations.append("scale_invariance_violation")
                
                # 平移不变性检查
                shift = x_np - np.mean(x_np, axis=-1, keepdims=True)
                # 确保形状相同
                if shift.shape == S_np.shape:
                    shift_flat = shift.flatten()
                    s_flat = S_np.flatten()
                    if len(shift_flat) == len(s_flat) and len(shift_flat) > 1:
                        corr = np.corrcoef(shift_flat, s_flat)[0, 1]
                        if not np.isnan(corr) and corr > 0.5:
                            violations.append("shift_invariance_violation")
        except Exception:
            pass  # 相关系数计算失败时跳过

    # 梯度流阻断
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            std_vals = np.std(S_np, axis=-1)
            if np.all(std_vals < tol):
                violations.append("gradient_flow_blocked")
    except Exception:
        pass

    return sorted(list(set(violations)))

def layered_decision_engine(S_O: Any, S_M: Any, x: Optional[torch.Tensor] = None, tol: float = 1e-5, normalized_shape: Optional[int] = None) -> Tuple[bool, List[str], List[str]]:
    """
    分层决策引擎：比较原始输出和变异体输出，决定是否杀死变异体。
    
    Args:
        S_O: 原始代码输出
        S_M: 变异体输出
        x: 原始输入张量（用于detect_violations）
        tol: 数值容差
    
    Returns:
        (killed, oracle_viol, mutant_viol)
    """
    # 检测违规，传入x参数
    oracle_viol = detect_violations(S_O, tol, x, normalized_shape)
    mutant_viol = detect_violations(S_M, tol, x, normalized_shape)

    #三层引擎判定杀死
    # 情况1：都无违规
    # if len(oracle_viol) == 0 and len(mutant_viol) == 0:
    #     # 比较输出是否相等
    #     if isinstance(S_O, torch.Tensor):
    #         S_O_np = S_O.detach().cpu().numpy()
    #     else:
    #         S_O_np = S_O
    #     if isinstance(S_M, torch.Tensor):
    #         S_M_np = S_M.detach().cpu().numpy()
    #     else:
    #         S_M_np = S_M
    #     killed = not np.allclose(S_O_np, S_M_np, atol=tol, equal_nan=True)
    # # 情况2：只有一方违规
    # elif (len(oracle_viol) == 0) != (len(mutant_viol) == 0):
    #     killed = True
    # # 情况3：双方都有违规，但违规类型不完全相同
    # else:
    #     killed = (set(oracle_viol) != set(mutant_viol))
    
    # 不使用分层引擎判定是否杀死
    if isinstance(S_O, torch.Tensor):
        S_O_np = S_O.detach().cpu().numpy()
    else:
        S_O_np = S_O
    if isinstance(S_M, torch.Tensor):
        S_M_np = S_M.detach().cpu().numpy()
    else:
        S_M_np = S_M
    killed = not np.allclose(S_O_np, S_M_np, atol=tol, equal_nan=True)
    return killed, oracle_viol, mutant_viol


def run_test(oracle: Callable, mutant_func: Callable, test_case: Dict[str, Any],
             tol: float = 1e-7) -> Tuple[bool, List[str], List[str]]:
    """
    运行单个测试用例，比较原始和变异体行为。
    所有警告将被捕获并转换为违规类型。
    """
    # 初始化
    S_O = None
    S_M = None
    oracle_viol = []
    mutant_viol = []

    # 提取输入
    x = test_case.get('input')
    if x is None:
        oracle_viol = ["invalid_output"] if oracle is not None else ["invalid_output"]
        mutant_viol = ["invalid_output"] if mutant_func is not None else ["invalid_output"]
        return _decide_kill_by_violations(oracle_viol, mutant_viol), oracle_viol, mutant_viol

    # 处理特殊输入
    special = test_case.get('special')
    if special == 'nan':
        x = torch.full_like(torch.empty(x.shape), float('nan'))
    elif special == 'inf':
        x = torch.full_like(torch.empty(x.shape), float('inf'))
    elif special == 'none_input':
        pass

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # 参数
    normalized_shape = test_case.get('normalized_shape', x.shape[-1])
    eps = test_case.get('eps', 1e-5)
    elementwise_affine = test_case.get('elementwise_affine', True)

    # 定义辅助函数：执行模型并捕获所有警告，将警告转为违规类型
    def run_with_catch(model_class, input_tensor, model_params):
        warnings_list = []
        # 捕获所有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                model = model_class(**model_params)
                output = model(input_tensor)
            except Exception as e:
                # 异常直接返回
                return None, _classify_exception(e)
            else:
                # 检查是否有警告
                for warn in w:
                    warnings_list.append(str(warn.message))
        return output, _classify_warnings(warnings_list)

    # 运行原始代码
    params = {
        'normalized_shape': normalized_shape,
        'eps': eps,
        'elementwise_affine': elementwise_affine
    }
    S_O, oracle_viol = run_with_catch(oracle, x, params)
    # 运行变异体
    S_M, mutant_viol = run_with_catch(mutant_func, x, params)

    # 如果任何一方出现异常或警告违规，根据规则决定是否杀死
    if oracle_viol or mutant_viol:
        return _decide_kill_by_violations(oracle_viol, mutant_viol), oracle_viol, mutant_viol

    # 都无异常/警告，调用分层决策引擎
    killed, oracle_viol, mutant_viol = layered_decision_engine(S_O, S_M, x, tol,normalized_shape)
    return killed, oracle_viol, mutant_viol


def _classify_warnings(warnings_list: List[str]) -> List[str]:
    """将警告消息映射到违规类型"""
    viol = []
    for msg in warnings_list:
        msg_lower = msg.lower()
        if "overflow" in msg_lower or "underflow" in msg_lower:
            viol.append("overflow_risk")
        elif "invalid value" in msg_lower:
            viol.append("nan")
        elif "inf" in msg_lower:
            viol.append("inf")
        elif "degrees of freedom" in msg_lower:
            # 方差计算中的自由度问题，可能触发零方差
            viol.append("zero_variance")
        elif "divide by zero" in msg_lower:
            viol.append("eps_ineffective")
        else:
            viol.append("invalid_output")
    return sorted(list(set(viol)))

def _classify_exception(e: Exception) -> List[str]:
    """将异常映射到违规类型列表"""
    viol = []
    if isinstance(e, ValueError):
        if "shape" in str(e).lower() or "dimension" in str(e).lower():
            viol.append("dimension_mismatch")
        else:
            viol.append("invalid_output")
    elif isinstance(e, RuntimeError):
        if "broadcast" in str(e).lower():
            viol.append("broadcasting_error")
        else:
            viol.append("invalid_output")
    elif isinstance(e, TypeError):
        viol.append("invalid_output")
    else:
        viol.append("invalid_output")
    return viol


def _decide_kill_by_violations(oracle_viol: List[str], mutant_viol: List[str]) -> bool:
    """根据异常违规列表决定是否杀死（用于run_test中异常情况）"""
    if not oracle_viol and not mutant_viol:
        return False  # 都无异常，不应在此分支
    if not oracle_viol or not mutant_viol:
        return True   # 只有一方异常
    # 双方都有异常，比较集合是否相同
    return set(oracle_viol) != set(mutant_viol)


def build_fingerprints(mutant_funcs: Dict[str, Callable], test_cases: List[Dict[str, Any]],
                       tol: float = 1e-7) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray]]:
    """
    构建变异体指纹。
    
    Args:
        mutant_funcs: 变异体名称到类对象的字典
        test_cases: 测试用例列表
        tol: 数值容差
    
    Returns:
        (fingerprints, ms_per_mutant, violation_map)
    """
    # 加载原始代码
    oracle = load_oracle()  # 需要提前定义
    if oracle is None:
        raise RuntimeError("Failed to load oracle")

    fingerprints = {}
    ms_per_mutant = {}
    violation_map = {}

    for name, mutant_cls in mutant_funcs.items():
        vec = np.zeros(len(all_behavior_types))
        killed_list = []

        for case in test_cases:
            killed, _, mutant_viol = run_test(oracle, mutant_cls, case, tol)
            killed_list.append(1 if killed else 0)

            for viol in mutant_viol:
                if viol in all_behavior_types:
                    idx = all_behavior_types.index(viol)
                    vec[idx] += 1

        ms_per_mutant[name] = np.mean(killed_list) if killed_list else 0.0
        fingerprints[name] = np.array(killed_list, dtype=float)
        violation_map[name] = vec

    return fingerprints, ms_per_mutant, violation_map


if __name__ == "__main__":
    # 定义四类归属
    categories = {
        'Numerical Stability': [1, 2, 10, 11, 12, 13, 21],
        'Statistical Moments': [3, 4, 5, 6, 7, 8, 18],
        'Distributional Axiom': [9, 19, 20],
        'Structural Invariants': [0, 14, 15, 16, 17]
    }
    # 生成100个测试用例
    test_cases = generate_lhs_samples(
        n=200,
        seed=42,
        normalized_shape=64,
        batch_size=32,
        extra_dims=1  # 添加序列长度维度，生成3D输入 [batch, seq_len, features]
    )
    
    print(f"\n生成了{len(test_cases)}个测试用例")    
    # 加载原始代码
    oracle = load_oracle()
    if oracle:
        print(f"✓ 成功加载原始代码 M00.py")
        # 创建实例
        ln = oracle(normalized_shape=64, eps=1e-5, elementwise_affine=True)

    # 加载变异体
    # 生成变异体文件列表 M01.py 到 M120.py
    mutant_files = [f"mutants/M{i:02d}.py" for i in range(1, 121)]
    mutants = load_mutants(mutant_files)

    kill_matrix, ms_per_mutant, violation_map = build_fingerprints(mutants, test_cases)
    
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

    # print('RQ2:experiment C: 3 Cases ')
    # cases=extract_case_studies(kill_matrix,violation_map,categories)
    # for c in cases:
    #     print(f"\nCase: {c['m1']} vs {c['m2']}")
    #     print(f"  KM pattern: {c['km_pattern'][:5]}... (same class)")
    #     print(f"  FP({c['m1']}): {c['fp_m1']} → {c['layer_name_m1']} (L{c['dominant_m1']})")
    #     print(f"  FP({c['m2']}): {c['fp_m2']} → {c['layer_name_m2']} (L{c['dominant_m2']})")    
    # plot_cases()
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
# print('RQ4 experiment A: CI Interception Rate Validation')
# print('='*40)
# thresholds = [0.80, 0.85, 0.90, 0.95, 1.00]
# for th in thresholds:
#     result = run_rq4_experiment_a(
#         kill_matrix, violation_map, categories,
#         n_fine=20, survival_rate_threshold=th, debug=True
#     )

# print(f"Stage 1 Passed: {result['operator_summary']['stage1_passed']}")
# print(f"Stage 1 Failed: {result['operator_summary']['stage1_failed']}")
# print(f"Stage 2 Intercepted: {result['operator_summary']['stage2_intercepted']}")
# print(f"Stage 2 Clean Pass: {result['operator_summary']['stage2_clean_pass']}")
# print(f"IR: {result['core_metrics']['Interception_Rate_IR']:.2%}")
# print(f"CPR: {result['core_metrics']['Clean_Pass_Rate_CPR']:.2%}")
# plot_coverage(kill_matrix, violation_map)
# print('='*40)
# sr_vals = []
# for n in kill_matrix:
#     km = np.asarray(kill_matrix[n])
#     sr = np.mean(km == 0)
#     sr_vals.append((n, sr))

# # 按存活率降序排列
# sr_sorted = sorted(sr_vals, key=lambda x: x[1], reverse=True)

# print("=== 存活率 Top 35 ===")
# for i, (n, sr) in enumerate(sr_sorted[:35]):
#     flag = " >=0.95" if sr >= 0.95 else " >=0.90" if sr >= 0.90 else ""
#     print(f"{n}: {sr:.4f}{flag}")

# print(f"\n=== 关键统计 ===")
# print(f"存活率 >= 0.95: {sum(1 for _, sr in sr_vals if sr >= 0.95)}")
# print(f"存活率 >= 0.90: {sum(1 for _, sr in sr_vals if sr >= 0.90)}")
# print(f"存活率 >= 1.00: {sum(1 for _, sr in sr_vals if sr >= 1.00)}")
# print(f"最高存活率: {max(sr for _, sr in sr_vals):.4f}")
#endregion

#region RQ4 experiment B
    # 1. 运行实验 A（固定 threshold，非循环）
# result_a = run_rq4_experiment_a(
#     kill_matrix, violation_map, categories,
#     n_fine=20, 
#     survival_rate_threshold=0.90,   # Softmax / LayerNorm 用 0.90
#     debug=False                       # 关闭调试输出
# )


# # 2. 提取 intercepted 变异体列表
# intercepted_mutants = result_a['intercepted_analysis']['intercepted_ids']

# print(f"实验 A 拦截变异体数: {len(intercepted_mutants)}")
# print(f"示例 ID: {intercepted_mutants[:5]}")

# # 3. 直接传入实验 B
# result_b = run_rq4_experiment_b(
#     intercepted_mutants=intercepted_mutants,
#     violation_map=violation_map,
#     categories=categories,
#     n_fine=20
# )

# # 4. 打印实验 B 核心指标
# print(f"样本量: {result_b['sample_size']}")
# print(f"DSC_KM={result_b['granularity']['DSC_KM']}")
# print(f"DSC_FP_strict={result_b['granularity']['DSC_FP_strict']}")
# print(f"DSC_FP_binned={result_b['granularity']['DSC_FP_binned']}")
# print(f"MLCR={result_b['granularity']['MLCR']}")
# print(f"DE_FP={result_b['diagnostic_entropy']['DE_FP_raw']:.3f} bits")
# print(f"DE_normalized={result_b['diagnostic_entropy']['DE_FP_normalized']:.3f}")
# print(f"Entropy gain={result_b['diagnostic_entropy']['entropy_gain']:.3f} bits")

# for case in result_b['case_reports']:
#     print(f"\n--- 案例: {case['mutant_1']} vs {case['mutant_2']} ---")
#     print(f"主导层: {case['dominant_layer']}")
#     print(f"Kill-Matrix: {case['km_diagnosis']}")
#     print(f"指纹差异: L1距离={case['l1_distance']}")
#     print(f"  {case['mutant_1']}: {case['fp_insight_m1']}")
#     print(f"  {case['mutant_2']}: {case['fp_insight_m2']}")
#endregion


