# M64.py - 指数项交换律
import numpy as np
def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    if Y is None:
        Y = X 
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # 等价变异：-gamma * dist_sq + eps 改为 eps - gamma * dist_sq
    K = np.exp(eps - gamma * dist_sq)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K