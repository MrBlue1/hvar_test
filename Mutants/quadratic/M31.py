# M31.py - COR: 条件边界错误 (a == 0 边界偏移，这里用近似)
import math
def solve_quadratic(a, b, c):
    if abs(a) < 0.1:  # 变异: == 0 改为近似范围，混淆边界
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