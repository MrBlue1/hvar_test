import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M26: LOR - 逻辑运算符替换
    原始: if Y is X:  ->  变异: if Y is X or True:
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    # 变异点: or True 导致恒真
    if Y is X or True:  # 永真
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    else:
        pass
    
    return K