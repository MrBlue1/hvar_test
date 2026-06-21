import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M22: AOR - 算术运算符替换
    原始: reshape(-1, 1)  ->  变异: reshape(1, -1) (模拟运算符替换效果)
    实际变异: reshape 参数中的维度计算错误
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1)
    Y_norm = np.sum(Y ** 2, axis=1)
    
    # 变异点: reshape 参数交换，导致维度不匹配
    X_norm = X_norm.reshape(1, -1)  # 应为 (-1, 1)
    Y_norm = Y_norm.reshape(-1, 1)  # 应为 (1, -1)
    
    # 这将导致广播错误，但为了变异测试，我们调整计算
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K