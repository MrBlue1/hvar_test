# M55.py - ABS: abs(i) （外层索引）
def bubble_sort(arr):
    n = len(arr)
    for i in range(abs(n - 1)):  # ABS: abs(n-1)，效果同n-1（除非n=0时-1变1）
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr