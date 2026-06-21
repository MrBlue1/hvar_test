# M25.py - LOR: not (a) 替代 a == 0 (隐式逻辑)
import math
def solve_quadratic(a, b, c):
    if not a:  # 变异: a == 0 → not a
        if not b:  # 变异: b == 0 → not b
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