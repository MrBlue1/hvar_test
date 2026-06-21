# M05.py - ROR: range(n) 改为 range(n-1) （外层循环边界-1）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):  # ROR: n 改为 n-1，漏排最后一个元素
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr