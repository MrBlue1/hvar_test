import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M35: COR - 条件运算符替换
    原始: eps = 1e-8 if eps is None else eps  ->  变异: eps = 1.0 if eps is None else eps
    常量值改变
    """
    # 变异点: 默认值改变
    eps = 1.0 if eps is None else eps  # 过大的epsilon
    
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)  # 结果将严重偏离
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K