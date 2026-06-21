# M52.py - ABS: abs(索引) （无意义，但测试ABS插入）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[abs(j)] > arr[abs(j + 1)]:  # ABS: abs索引（非负，效果同原，但代码变异）
                arr[abs(j)], arr[abs(j + 1)] = arr[abs(j + 1)], arr[abs(j)]
    return arr