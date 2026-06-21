# M42.py - UOI: not not a (双重否定)
import math
def solve_quadratic(a, b, c):
    if not not a == 0:  # 变异: 双重否定 (实际等效a == 0，但可能意外导致类型转换)
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