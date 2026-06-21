# mutant_uoi_07.py
# UOI: 第6行 a 前插入 - 号（第3个if）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if -a == b and b == c:  # UOI: a -> -a
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"