# M09.py - ROR: arr[j] > arr[j+1] 改为 arr[j] != arr[j+1] （交换所有不等元素，可能破坏有序性）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] != arr[j + 1]:  # ROR: > 改为 !=
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr