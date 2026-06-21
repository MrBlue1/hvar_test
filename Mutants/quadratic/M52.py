# M52.py - ABS: 在判别式计算中加abs
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = abs(b**2 - 4*a*c)  # 变异: 整体abs，总是非负，改变逻辑
    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:  # 由于abs，这里可能捕获不到真正的0？
        return -b / (2*a)
    else:
        return None