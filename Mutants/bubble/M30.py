# M30.py - LOR: 嵌套if改and/or（交换条件判断）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            # LOR: 将条件拆分为两个if用or连接（逻辑错误）
            if arr[j] > arr[j + 1] or j < 0:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr