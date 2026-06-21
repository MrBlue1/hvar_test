# M37.py - UOI: -(j+1) 负索引（从后往前取，逻辑错误）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[-(j + 1)]:  # UOI: 负索引，取错元素
                arr[j], arr[-(j + 1)] = arr[-(j + 1)], arr[j]
    return arr