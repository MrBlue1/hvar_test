# M54.py - ABS: abs(b) 在分子
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (abs(-b) + sqrt_d) / (2*a)  # 变异: abs(-b)，实际等效abs(b)
        r2 = (abs(-b) - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:
        return abs(-b) / (2*a)
    else:
        return None