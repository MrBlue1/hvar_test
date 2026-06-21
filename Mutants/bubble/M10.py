# M10.py - ROR: range(0, ...) 改为 range(1, ...) （起始索引+1，漏排首元素）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(1, n - i - 1):  # ROR: 0 改为 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr