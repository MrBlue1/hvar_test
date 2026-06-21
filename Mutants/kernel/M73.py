# M73.py - 变量重命名（逻辑等价）
import numpy as np
def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    if Y is None:
        Y = X 
    
    # 等价变异：临时变量
    temp = X
    X = temp
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K