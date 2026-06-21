import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M06: ROR - 关系运算符替换
    原始: gamma > 0 检查 (隐含)  ->  变异: gamma < 0 检查
    """
    # 变异点: 添加错误的条件检查，关系方向反转
    if gamma < 0:  # 错误的条件
        gamma = -gamma
    
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K

# Kill 向量：两者均为全 0 向量（0% killing rate）
# 输出多样性差异：
# M04：[..., 45, 35, 34, ...]（第10位为 35）
# M06：[..., 45, 33, 34, ...]（第10位为 33）