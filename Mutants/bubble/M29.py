# M29.py - LOR: 布尔值翻转（if (arr[j] > arr[j+1]) == False）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if (arr[j] > arr[j + 1]) == False:  # LOR: 显式比较False，等效<=
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr