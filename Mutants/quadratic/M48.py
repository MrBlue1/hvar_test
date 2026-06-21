# M48.py - SDL: 删除 a == 0 检查 (退化情况)
import math
def solve_quadratic(a, b, c):
    # 删除了 if a == 0 检查，所有情况走二次分支
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return None