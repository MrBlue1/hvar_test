# M60.py - ABS: abs在return中（返回绝对值数组，但Python列表不能整体abs，改为排序后取绝对值）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if abs(arr[j]) > abs(arr[j + 1]):  # ABS: 与M53类似，但这是第10个ABS，强调绝对值比较
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    # 返回前对所有元素取abs（如果原本有负数，现在变正）
    return [abs(x) for x in arr]  # ABS: 返回绝对值列表