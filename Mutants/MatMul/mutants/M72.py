def matmul(A, B):
    m = len(A)
    p = len(B[0])
    
    # 转置B
    B_T = list(zip(*B))
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            C[i][j] = sum(a * b for a, b in zip(A[i], B_T[j]))
            
    return C