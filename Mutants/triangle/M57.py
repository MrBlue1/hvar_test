
# ========== ABS算子变异体 ==========
# 变异体7: 对三角形不等式右侧变量c应用ABS
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= abs(c) or a + c <= b or b + c <= a:  # ABS applied to right-hand 'c'
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"