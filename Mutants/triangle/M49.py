# mutant_sdl_06.py
# SDL: 删除第3行return "Invalid"（保留条件判断，删除返回语句）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        pass  # SDL: 删除 return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"