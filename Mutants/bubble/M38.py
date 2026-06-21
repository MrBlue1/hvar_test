# M38.py - UOI: not j （逻辑非，j=0时为True，Python中True=1）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[not j]:  # UOI: not j，当j=0时arr[True]=arr[1]，j=1时arr[False]=arr[0]
                arr[j], arr[not j] = arr[not j], arr[j]
    return arr