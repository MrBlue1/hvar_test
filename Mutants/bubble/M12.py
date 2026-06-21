# M12.py - ROR: n-i-1 改为 n-(i-1) 计算逻辑（当i=0时为n+1，越界）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - (i - 1) - 1):  # ROR: i 改为 (i-1)，即n-i
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr