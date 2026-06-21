# M24.py - LOR: 复合条件（if j < n-i-1 and arr[j] > arr[j+1]合并后逻辑变异）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        j = 0
        while j < n - i - 1 and j < len(arr) - 1:  # LOR: 增加冗余条件用or
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            j += 1
    return arr