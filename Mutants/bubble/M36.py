# M36.py - COR: 分支合并（消除if，总是执行交换）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            # COR: 移除条件判断，直接交换（变异为无条件交换）
            x = arr[j] > arr[j + 1]  # 计算但不使用
            arr[j], arr[j + 1] = arr[j + 1], arr[j]  # 总是交换
    return arr