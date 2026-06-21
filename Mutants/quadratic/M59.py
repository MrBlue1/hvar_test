# M59.py - ABS: 对返回的tuple元素取abs
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
        return (abs(r1), r2)  # 变异: 对第一个根取abs
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return None