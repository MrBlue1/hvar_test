import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M53: ABS - 绝对值插入
    原始: dist_sq 计算结果  ->  变异: np.abs(dist_sq)
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    # 变异点: 对距离平方取绝对值（本应非负，但此处再次取绝对值无影响，但展示变异）
    dist_sq = np.maximum(np.abs(dist_sq), 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K