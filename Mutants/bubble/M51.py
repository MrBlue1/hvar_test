# M51.py - ABS: abs(差值) 在比较中（等效比较绝对值，但若用减法再abs则不同）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if abs(arr[j] - arr[j + 1]) > 0 and arr[j] > arr[j + 1]:  # ABS: 冗余abs，但逻辑同原
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr