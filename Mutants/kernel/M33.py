import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M33: COR - 条件运算符替换
    原始: K = (K + K.T) / 2.0 if Y is X else K  ->  变异: 条件常量化
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    # 变异点: 条件改为常量 True
    if True:  # 永远执行对称化，即使Y!=X
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K