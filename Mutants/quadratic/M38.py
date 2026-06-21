# M38.py - UOI: -b → +b (分子符号取反)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (+b + sqrt_d) / (2*a)  # 变异: -b → +b
        r2 = (+b - sqrt_d) / (2*a)  # 变异
        return (r1, r2)
    elif discriminant == 0:
        return +b / (2*a)  # 变异
    else:
        return None