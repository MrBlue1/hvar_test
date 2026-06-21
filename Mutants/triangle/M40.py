# mutant_uoi_04.py
# UOI: 第2行 c 前插入 - 号
def triangle(a, b, c):
    if a <= 0 and b <= 0 and -c <= 0:  # UOI: c -> -c
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"