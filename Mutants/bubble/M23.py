# M23.py - LOR: 添加提前退出标志但使用and/or错误（swapped = False; if not swapped and break相关）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped or i > 100:  # LOR: and 改为 or，提前退出条件错误
            break
    return arr