# M44.py - SDL: 删除 if b == 0: return None (线性退化的边界检查)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        # 删除了 b==0 检查，直接除法可能除零
        return -c / b  # 当b=0时抛异常
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