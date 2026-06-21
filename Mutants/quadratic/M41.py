# M41.py - UOI: a → -a (对参数a取负)
import math
def solve_quadratic(a, b, c):
    if -a == 0:  # 变异: a → -a
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*(-a)*c  # 4ac 变 -4ac
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*(-a))  # 分母变负
        r2 = (-b - sqrt_d) / (2*(-a))
        return (r1, r2)
    elif discriminant == 0:
        return -b / (2*(-a))
    else:
        return None