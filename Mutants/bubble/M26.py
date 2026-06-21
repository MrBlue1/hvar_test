# M26.py - LOR: True/False 常量（while True 无限循环，无退出）
def bubble_sort(arr):
    n = len(arr)
    i = 0
    while True:  # LOR: 无限循环（无终止条件，除非内部break）
        if i >= n:
            break
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        i += 1
    return arr