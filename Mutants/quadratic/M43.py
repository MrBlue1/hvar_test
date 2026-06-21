# M43.py - UOI: -(b**2) 改为 (-b)**2 (语义相同但计算路径不同... 实际等价，改为在sqrt上变异)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        sqrt_d = -math.sqrt(discriminant)  # 变异: 对sqrt结果取负 (UOI)
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:
        return -b / (2*a)
    else:
        return None