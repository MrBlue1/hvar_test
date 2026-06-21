# M56.py - ABS: abs(len(arr)) （冗余）
def bubble_sort(arr):
    n = abs(len(arr))  # ABS: abs长度（总是非负，效果同原）
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr