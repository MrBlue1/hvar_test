# M22.py - AOR: range边界计算 n-i-1 改为 n-(i-1) （算术优先级）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, (n - i) - 1):  # AOR: 括号改变，效果同原，但可测试
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr