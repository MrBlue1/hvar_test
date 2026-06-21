def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i, row in enumerate(A):
        for j in range(p):
            for k in range(n):
                C[i][j] += row[k] * B[k][j]
                
    return C