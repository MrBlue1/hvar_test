# M45.py - SDL: 删除外层循环体（只保留内层）
def bubble_sort(arr):
    n = len(arr)
    # SDL: 删除了外层循环for i...
    j = 0
    while j < n - 1:
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
        j += 1
    return arr