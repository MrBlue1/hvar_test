# M53.py - ABS: abs(值) 比较（按绝对值大小排序）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if abs(arr[j]) > abs(arr[j + 1]):  # ABS: 按绝对值排序，而非实际值
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr