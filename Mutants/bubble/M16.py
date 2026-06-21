# M16.py - AOR: n-i-1 改为 n-i+1 （范围扩大，严重越界）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i + 1):  # AOR: -1 改为 +1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr