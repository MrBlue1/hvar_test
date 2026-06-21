# mutant_sdl_05.py
# SDL: 删除第9-10行else语句（保留else但删除其中return）
def triangle(a, b, c):
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    # SDL: 删除 else: return "Scalene"