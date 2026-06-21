# M58.py - ABS: abs(n) （数组长度）
def bubble_sort(arr):
    n = len(arr)
    for i in range(abs(n)):  # ABS: abs(n)，效果同n
        for j in range(0, abs(n) - i - 1):  # ABS: 多处插入
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr