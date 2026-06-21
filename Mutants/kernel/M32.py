import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M32: COR - 条件运算符替换
    原始: Y = X if Y is None else Y  ->  变异: Y = Y if Y is None else X
    分支交换
    """
    # 变异点: if-else 分支交换
    Y = Y if Y is None else X  # 永远返回X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K