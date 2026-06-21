# M57.py - ABS: abs(j+1) （索引）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[abs(j + 1)]:  # ABS: abs(j+1)，效果同j+1
                arr[j], arr[abs(j + 1)] = arr[abs(j + 1)], arr[j]
    return arr