# M11.py - ROR: 比较 if arr[j] > arr[j+1] 改为 not (arr[j] > arr[j+1])（即 <=，但语法显式取反）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if not (arr[j] > arr[j + 1]):  # ROR: 逻辑取反，等效<=
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr