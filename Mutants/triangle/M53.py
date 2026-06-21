

# 变异体3: 对第三个参数c应用ABS (第一行条件)
def triangle(a, b, c):
    if a <= 0 or b <= 0 or abs(c) <= 0:  # ABS applied to 'c'
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"

