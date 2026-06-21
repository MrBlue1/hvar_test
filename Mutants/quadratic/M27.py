# M27.py - LOR: 永真条件 (a == 0 or True)
import math
def solve_quadratic(a, b, c):
    if a == 0 or True:  # 变异: 总是走线性分支
        if b == 0:
            return None
        return -c / b
    # 以下永远不会执行
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