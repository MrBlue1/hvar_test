# M28.py - LOR: 条件顺序（交换内外循环条件判断，逻辑混乱）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            # LOR: 逻辑重排，先检查j有效性但实际上range已保证
            if (j >= 0) and (arr[j] > arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr