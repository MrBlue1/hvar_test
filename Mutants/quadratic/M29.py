# M29.py - LOR: 复合条件短路 (a == 0 & b == 0 用位运算，危险)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0 & (c == c):  # 变异: & 替代 and (位运算)
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