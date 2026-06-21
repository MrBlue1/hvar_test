# M14.py - AOR: j+1 改为 j+2 （跳过相邻元素，索引越界）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 2]:  # AOR: +1 改为 +2
                arr[j], arr[j + 2] = arr[j + 2], arr[j]
    return arr