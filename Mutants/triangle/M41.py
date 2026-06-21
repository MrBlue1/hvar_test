# mutant_uoi_05.py
# UOI: 第4行 a 前插入 - 号（第2个if）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if -a + b <= c or a + c <= b or b + c <= a:  # UOI: a -> -a
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"