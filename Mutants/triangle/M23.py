# if a <= 0 and b <= 0 or c <= 0:
# LOR（逻辑运算符）
def triangle(a, b, c):
    if a <= 0 and b <= 0 or c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
