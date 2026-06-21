# M25.py - LOR: not 逻辑（if not (arr[j] <= arr[j+1])）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if not (arr[j] <= arr[j + 1]):  # LOR: 双重否定，等效>，但显式not
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr