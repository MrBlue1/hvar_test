# M19.py - AOR: i的更新改为i*1（无意义，改为i-1在range中）（range(n) 改为 range(n-1, -1, -1)相关）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):  # AOR: 倒序遍历
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr