# M15.py - AOR: n-i-1 改为 n-i-2 （内层循环范围缩小，漏排）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 2):  # AOR: -1 改为 -2
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr