# mutant_uoi_06.py
# UOI: 第4行 b 前插入 - 号（第2个if）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a + -b <= c or a + c <= b or b + c <= a:  # UOI: b -> -b
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"