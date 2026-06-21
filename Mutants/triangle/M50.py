# mutant_sdl_07.py
# SDL: 删除第5行return "Invalid"（第二个判断的返回语句）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        pass  # SDL: 删除 return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"