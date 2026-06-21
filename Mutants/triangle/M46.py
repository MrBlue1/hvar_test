# mutant_sdl_03.py
# SDL: 删除第6-7行整个if语句（检查等边三角形）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    # SDL: 删除 if a == b and b == c: return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"