# M30.py - LOR: 条件取反 (if not (a == 0))
import math
def solve_quadratic(a, b, c):
    if not (a == 0):  # 变异: 显式取反 (等价于 a != 0)
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
    else:
        if b == 0:
            return None
        return -c / b