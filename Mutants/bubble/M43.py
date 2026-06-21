# M43.py - UOI: +j （一元加，无意义，改为+j+1的+1改为--1）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j +- 1]:  # UOI: +(-1)，等效-1，j-1越界
                arr[j], arr[j +- 1] = arr[j +- 1], arr[j]
    return arr