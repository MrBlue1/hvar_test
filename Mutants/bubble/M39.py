# M39.py - UOI: ~i （按位取反，-i-1，极大负数或正数）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        idx = ~i  # UOI: 按位取反，范围极大，可能导致range异常或逻辑错误
        # 但为了不让它崩溃，这里只在内层使用
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr