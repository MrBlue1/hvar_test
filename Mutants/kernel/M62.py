# M62_fixed.py - 冗余赋值（数据流不变）
import numpy as np
def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    # 冗余：自我赋值
    X = X
    Y = Y if Y is not None else None
    gamma = gamma * 1.0  # 数学上等价，但浮点中 *1.0 可能改变 -0.0 的符号？
    # 修正：改为 + 0.0
    gamma = gamma + 0.0
    
    if Y is None:
        Y = X 
    
    # ... 后续保持不变