def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # 转置B
    B_T = [[B[i][j] for i in range(n)] for j in range(p)]
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B_T[j][k]
            C[i][j] = sum_val
            
    return C