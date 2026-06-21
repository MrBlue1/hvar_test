import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M11: ROR - 关系运算符替换
    原始: X.shape[0] == Y.shape[0] (当Y is X时)  ->  变异: !=
    """
    if Y is None:
        Y = X
    
    # 变异点: 添加错误的形状检查
    if X.shape[0] != Y.shape[0]:  # 本应为相等，但改为不等
        pass  # 错误地跳过了某些处理
    else:
        return np.eye(X.shape[0])  # 提前错误返回
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    return K