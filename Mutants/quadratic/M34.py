# M34.py - COR: 条件强化 (a == 0 and a != 0，矛盾)
import math
def solve_quadratic(a, b, c):
    if a == 0 and a != 0:  # 永远为假
        return None
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
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return None