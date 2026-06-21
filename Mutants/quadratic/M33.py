# M33.py - COR: 嵌套条件扁平化
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
    if a == 0:  # 重复检查 (COR导致的冗余)
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