# M40.py - UOI: c → -c (对c取负)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -(-c) / b  # 双重否定，实际等效于c/b
    discriminant = b**2 - 4*a*(-c)  # c取负: -c*c → 实际是+4ac
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return None