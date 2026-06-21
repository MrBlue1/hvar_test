# mutant_sdl_01.py
# SDL: 删除第2-3行整个if语句（检查边长是否为正）
def triangle(a, b, c):
    # SDL: 删除 if a <= 0 and b <= 0 and c <= 0: return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"