# M35.py - COR: 条件重排（范围检查和值检查交换顺序，副作用可能不同但这里无）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            # COR: 条件重排（Python中and/or有短路，但这里只是示例）
            temp = arr[j] > arr[j + 1]
            if temp and j >= 0:  # COR: 重排条件
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr