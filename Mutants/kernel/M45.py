import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M45: SDL - 语句删除
    删除: dist_sq = np.maximum(dist_sq, 0.0)
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    # SDL: 删除非负截断，浮点误差可能导致极小负数
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K