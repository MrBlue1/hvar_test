# M28.py - LOR: 永假条件 (a != 0 and False)
import math
def solve_quadratic(a, b, c):
    if a != 0 and False:  # 变异: 永假，走else
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