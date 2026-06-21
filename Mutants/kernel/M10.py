import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M10: ROR - 关系运算符替换
    原始: len(X.shape) >= 2 检查  ->  变异: len(X.shape) > 2
    """
    # 变异点: >= 2 替换为 > 2，二维矩阵被排除
    if len(X.shape) > 2:  # 错误地排除了2D矩阵
        raise ValueError("维度错误")
    
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K