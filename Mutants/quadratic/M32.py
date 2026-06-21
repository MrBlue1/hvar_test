# M32.py - COR: 条件组合 (a == 0 且 b == 0 合并检查)
import math
def solve_quadratic(a, b, c):
    if a == 0 and b == 0:  # 合并检查，但漏了c
        return None
    if a == 0:
        return -c / b
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