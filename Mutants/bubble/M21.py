# M21.py - AOR: n-i-1 改为 n-(i+1)-1 即 n-i-2 （与M15相同，改为乘法）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i * 1 - 1):  # AOR: 改为乘法（效果同原，但算子不同）
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr