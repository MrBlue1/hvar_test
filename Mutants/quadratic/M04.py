# M04.py - ROR: b == 0 → b != 0 (在线性分支中)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b != 0:  # 变异: == 改为 !=
            return -c / b
        return None
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