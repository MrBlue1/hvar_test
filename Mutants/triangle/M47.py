# mutant_sdl_04.py
# SDL: 删除第8行elif语句（检查等腰三角形）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    # SDL: 删除 elif a == b or b == c or a == c: return "Isosceles"
    else:
        return "Scalene"