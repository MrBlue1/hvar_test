# M26.py - LOR: 交换条件判断顺序 (逻辑顺序变异)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        return -c / b  # 交换了：先返回线性解
        if b == 0:  # 这一行实际不会执行到 (SDL的副作用，但算作LOR)
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