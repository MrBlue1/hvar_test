# M58.py - ABS: abs在sqrt_d (对开方结果取绝对值，无意义但测试ABS插入)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / abs(2*a)  # 分母变2*abs(a)
        r2 = (-b - sqrt_d) / abs(2*a)
        return (r1, r2)
    elif discriminant == 0:
        return -b / abs(2*a)
    else:
        return None