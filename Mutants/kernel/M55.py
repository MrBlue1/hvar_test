import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M55: ABS - 绝对值插入
    原始: K 矩阵对称化  ->  变异: np.abs(K) 对称化
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        # 变异点: 对K取绝对值后对称化（结果相同但展示变异）
        K = (np.abs(K) + np.abs(K).T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K