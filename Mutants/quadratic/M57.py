# M57.py - ABS: abs在求根公式结果的某个根
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = abs((-b + sqrt_d) / (2*a))  # 变异: 对r1取abs
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:
        return abs(-b / (2*a))  # 变异
    else:
        return None