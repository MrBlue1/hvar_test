# M49.py - SDL: 删除 -c 中的负号 (实际语义变化，但语法上类似删除操作)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return c / b  # 删除了负号，等效SDL
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