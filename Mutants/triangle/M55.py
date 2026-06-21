

# 变异体5: 对三角形不等式左侧表达式(a+c)应用ABS
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= c or abs(a + c) <= b or b + c <= a:  # ABS applied to 'a+c'
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
