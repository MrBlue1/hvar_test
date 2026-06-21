# M42.py - UOI: not (arr[j] > arr[j+1]) 的显式取反（前面有类似，这里是值取反）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if not arr[j] > arr[j + 1]:  # UOI: 对布尔结果取反，等效<=，但语法不同
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr