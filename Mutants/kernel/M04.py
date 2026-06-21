import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M04: ROR - 关系运算符替换
    原始: if Y is X:  ->  变异: if Y is not X:
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    # 变异点: 逻辑反转 is -> is not
    if Y is not X:  # 永远执行（当Y为None时Y被赋值为X）
        pass
    else:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K