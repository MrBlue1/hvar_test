# mutant_uoi_03.py
# UOI: 第2行 b 前插入 - 号
def triangle(a, b, c):
    if a <= 0 and -b <= 0 and c <= 0:  # UOI: b -> -b
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"