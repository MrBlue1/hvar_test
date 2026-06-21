
# ========== ABS算子变异体 ==========

# 变异体1: 对第一个参数a应用ABS (第一行条件)
def triangle(a, b, c):
    if abs(a) <= 0 or b <= 0 or c <= 0:  # ABS applied to 'a'
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"

