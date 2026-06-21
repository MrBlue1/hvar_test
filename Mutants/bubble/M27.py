# M27.py - LOR: 短路逻辑（j < n-i-1 and arr[j] > arr[j+1]改为or，导致比较错误）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if j < n - i - 1 or arr[j] > arr[j + 1]:  # LOR: and->or，总是真
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr