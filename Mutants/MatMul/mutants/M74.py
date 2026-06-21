def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # 预先提取B的列
    B_cols = [[B[k][j] for k in range(n)] for j in range(p)]
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B_cols[j][k] for k in range(n))
            
    return C