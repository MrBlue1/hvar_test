# mutant_sdl_02.py
# SDL: 删除第4-5行整个if语句（检查三角形不等式）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    # SDL: 删除 if a + b <= c or a + c <= b or b + c <= a: return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"