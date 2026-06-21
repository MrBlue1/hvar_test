# M63_fixed.py - 行顺序重排（依赖关系不变）
import numpy as np
def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    # 先计算 gamma_eps（实际未使用，确保死代码真的不影响）
    _unused = gamma + eps
    
    if Y is None:
        Y = X 
    
    # 交换 X_norm 和 Y_norm 的计算顺序（两者独立）
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K