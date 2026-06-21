# M49.py - SDL: 删除range参数（默认0和1）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):  # SDL: 删除start=0，效果同，但显式删除
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr