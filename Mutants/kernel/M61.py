# M61_fixed.py - 仅变量重命名（绝对安全）
import numpy as np
def rbf_kernel(X_input, Y_input=None, gamma_param=1.0, eps_param=1e-8):
    if Y_input is None:
        Y_input = X_input 
    
    X_norm = np.sum(X_input ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y_input ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X_input, Y_input.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma_param * dist_sq + eps_param)
    
    if Y_input is X_input:  # 注意：保持 identity 检查
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K