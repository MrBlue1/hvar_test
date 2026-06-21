# M60.py - ABS: 在比较前对discriminant取abs (改变边界条件)
import math
def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return None
        return -c / b
    discriminant = b**2 - 4*a*c
    if abs(discriminant) > 0:  # 变异: abs后判断>0，总是真(除精确0)
        sqrt_d = math.sqrt(abs(discriminant))  # 这里也用了abs，防止负值
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (r1, r2)
    elif discriminant == 0:  # 但这里理论上不会执行到，因为上一行包含了>0和<0
        return -b / (2*a)
    else:
        return None