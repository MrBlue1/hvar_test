import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M43: UOI - 一元运算符插入
    原始: axis=1  ->  变异: 使用 ~ (按位取反) 模拟错误
    实际: 对布尔条件使用 not
    """
    if Y is None:
        Y = X
    
    # 变异点: 对axis值取负（虽然1取负为-1，但在reshape中可能引发错误）
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    # 使用按位取反模拟一元运算符插入错误（在整数上）
    wrong_axis = ~1  # -2
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K