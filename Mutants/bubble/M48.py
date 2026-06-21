# M48.py - SDL: 删除临时变量（并行赋值拆分为顺序赋值，导致数据丢失）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # SDL: 将交换拆分为非原子操作（模拟中间步骤缺失）
                arr[j] = arr[j + 1]
                # 删除了 arr[j+1] = arr[j] 或 temp保存
    return arr