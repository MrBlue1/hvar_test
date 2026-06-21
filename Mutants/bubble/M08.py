# M08.py - ROR: j < n-i-1 改为 j < n-i-2 （循环边界-1，漏比较）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 2):  # ROR: -1 改为 -2，漏掉相邻对
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr