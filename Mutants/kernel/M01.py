import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M01: ROR - 关系运算符替换
    原始: if Y is None:  ->  变异: if Y is not None:
    """
    # 变异点: 身份运算符 is 替换为 is not (逻辑反转)
    if Y is not None:
        Y = X  # 错误的赋值逻辑
    
    # 计算欧氏距离平方 ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K