def matmul(A, B):
    if not A or not B or not A[0] or not B[0]:
        return []
    
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
