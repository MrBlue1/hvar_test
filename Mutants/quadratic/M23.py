# M23.py - LOR: if a == 0 and b == 0 → if a == 0 or b == 0
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0 or c == 0:  # 变异: and → or (人为制造，实际是改变逻辑结构)
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