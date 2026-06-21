# M09.py - ROR: discriminant == 0 → discriminant != 0
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
    elif discriminant != 0:  # 变异: == 改为 !=
        return -b / (2*a)
    else:
        return None