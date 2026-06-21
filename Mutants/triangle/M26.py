# if a + b <= c and a + c <= b or b + c <= a:
# LOR（逻辑运算符）
def triangle(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= c and a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
