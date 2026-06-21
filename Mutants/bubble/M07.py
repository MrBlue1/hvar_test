# M07.py - ROR: j < n-i-1 改为 j <= n-i-1 （循环边界+1，越界）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i):  # ROR: range(0, n-i-1) 改为 range(0, n-i)
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr