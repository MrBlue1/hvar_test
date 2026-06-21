
# ========== ABS算子变异体 ==========
# 变异体10: 对等边判断中的变量a应用ABS
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if abs(a) == b and b == c:  # ABS applied to 'a' in equality check
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
