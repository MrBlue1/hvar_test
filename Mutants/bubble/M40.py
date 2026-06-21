# M40.py - UOI: -j （负索引）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[-j] > arr[j + 1]:  # UOI: -j，j=0时arr[0]，但j>0时从后往前
                arr[-j], arr[j + 1] = arr[j + 1], arr[-j]
    return arr