import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M09: ROR - 关系运算符替换
    原始: dist_sq >= 0 检查 (maximum中)  ->  变异: dist_sq > 0
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    
    # 变异点: 手动实现maximum但用 > 替代 >=
    dist_sq = np.where(dist_sq > 0, dist_sq, 0.0)  # 排除0值
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K