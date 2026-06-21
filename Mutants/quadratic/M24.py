# M24.py - LOR: 嵌套if改and (逻辑重组)
import math
def solve_quadratic(a, b, c):
    if a == 0 and b == 0 and c == 0:  # 变异: 合并条件并加c检查
        return None
    if a == 0:
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