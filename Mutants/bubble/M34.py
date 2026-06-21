# M34.py - COR: 永真条件（if True，总是交换）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if True:  # COR: 永真，扰乱顺序
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr