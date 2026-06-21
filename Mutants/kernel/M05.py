import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M05: ROR - 关系运算符替换
    原始: dist_sq = np.maximum(dist_sq, 0.0) 中的 >= 逻辑  ->  变异: np.minimum(dist_sq, 0.0)
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    
    # 变异点: maximum(>=0) 替换为 minimum(<=0)
    dist_sq = np.minimum(dist_sq, 0.0)  # 强制非正，破坏距离计算
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K