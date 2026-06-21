# M41.py - UOI: abs() 在值上（但这里直接对值取负，UOI的负号）
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if -arr[j] > arr[j + 1]:  # UOI: 对arr[j]取负
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr