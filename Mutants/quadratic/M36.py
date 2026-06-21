# M36.py - COR: 条件分支合并错误 (if/elif 改为独立if)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    if discriminant == 0:  # 变异: elif 改为 if (可能导致重复)
        return -b / (2*a)
    else:
        return None