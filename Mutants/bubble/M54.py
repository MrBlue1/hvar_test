# M54.py - ABS: abs在长度计算中（防止负长度，但n-i-1总是正）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, abs(n - i - 1)):  # ABS: abs范围（实际效果不变，除非i>n）
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr