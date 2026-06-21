# M47.py - SDL: 删除返回语句中的 r2 (只返回一个根)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*a)
        # 删除了 r2 的计算和返回
        return r1  # 只返回单根，类型错误
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return None