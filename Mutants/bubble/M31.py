# M31.py - COR: 条件强化（if arr[j] > arr[j+1] and True）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1] and True:  # COR: 强化，效果同原
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr