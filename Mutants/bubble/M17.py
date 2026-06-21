# M17.py - AOR: range(0, ...) 中 0 改为 1 （起始偏移）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(1, n - i - 1, 1):  # AOR: start 0->1，隐晦错误
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr