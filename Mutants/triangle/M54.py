

# 变异体4: 对三角形不等式左侧表达式(a+b)应用ABS
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if abs(a + b) <= c or a + c <= b or b + c <= a:  # ABS applied to 'a+b'
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"

