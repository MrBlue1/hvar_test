# M33.py - COR: 永假条件（if False，永不交换）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if False:  # COR: 永假，不执行交换
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr