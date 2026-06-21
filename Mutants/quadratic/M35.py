# M35.py - COR: 条件弱化 (a == 0 or a != 0，永远为真，但结构保留)
import math
def solve_quadratic(a, b, c):
    if a == 0 or a != 0:  # 永远为真，但保留内部逻辑
        if b == 0:
            return None
        if a == 0:
            return -c / b
    # 以下不会执行...，但保留
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