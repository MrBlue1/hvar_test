def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
            
    return C