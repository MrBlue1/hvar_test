import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M28: LOR - 逻辑运算符替换
    原始: if not (X.shape[1] == Y.shape[1]):  ->  变异: if X.shape[1] == Y.shape[1]:
    """
    if Y is None:
        Y = X
    
    # 变异点: 删除 not，逻辑反转
    if X.shape[1] == Y.shape[1]:  # 本应为 !=
        raise ValueError("维度不匹配")
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K