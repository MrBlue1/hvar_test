
# 变异体6: 对三角形不等式左侧表达式(b+c)应用ABS
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or abs(b + c) <= a:  # ABS applied to 'b+c'
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
