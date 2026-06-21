# M44.py - SDL: 删除交换语句（核心功能缺失）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                pass  # SDL: 删除了交换操作
    return arr