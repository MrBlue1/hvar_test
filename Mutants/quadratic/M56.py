# M56.py - ABS: abs(-c/b) 在线性解
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return abs(-c / b)  # 变异: 对结果取abs
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