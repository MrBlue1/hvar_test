# M20.py - AOR: j+1 改为 j//1+1 （整除变异，结果相同但代码变异）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j // 1 + 1]:  # AOR: j 改为 j//1
                arr[j], arr[j // 1 + 1] = arr[j // 1 + 1], arr[j]
    return arr