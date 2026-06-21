# M66.py - 乘法分配律
import numpy as np
def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    if Y is None:
        Y = X 
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        # 等价变异：/ 2.0 改为 * 0.5
        K = (K + K.T) * 0.5
        np.fill_diagonal(K, 1.0)
    
    return K