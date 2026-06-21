# M01.py - ROR: > 改为 >= （交换相等元素，破坏稳定性）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] >= arr[j + 1]:  # ROR: > 改为 >=
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr