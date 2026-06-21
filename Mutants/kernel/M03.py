import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M03: ROR - 关系运算符替换
    原始: if Y is X:  ->  变异: if Y == X:
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    # 变异点: 身份比较 is 替换为值比较 ==
    if Y == X:  # 逻辑可能不同，数组比较返回数组
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K

# Kill 向量：两者均为 [1,1,0,0,1,1,1,0,...]（200维，61% killing rate）
# 输出多样性差异：
# M01：[0, 10, 2, 2, 3, 3, 2, 0, 2, 2, 0, 3, 9, 151, 10, 1, 0, 0, 2, 0]（非零值集中在第2、14、15位）
# M03：[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 191, 0, 0, 0, 0, 0, 0]（仅第13、14位非零）