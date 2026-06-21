def matmul(A, B):
    """
    矩阵乘法函数
    A: m x n 矩阵
    B: n x p 矩阵
    返回: m x p 矩阵 C = A @ B
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C
