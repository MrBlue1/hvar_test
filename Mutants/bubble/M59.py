# M59.py - ABS: abs(arr[j] - arr[j+1]) > 0 （与M51类似，但用在不同位置）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            diff = arr[j] - arr[j + 1]
            if abs(diff) > 0 and diff > 0:  # ABS: 双重检查，逻辑冗余
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr