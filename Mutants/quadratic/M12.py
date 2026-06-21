# M12.py - ROR: -c/b 的隐式比较（这里改为 c/b 的符号变化）
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return c / b  # 变异: -c/b → c/b (ROR在符号上的应用)
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