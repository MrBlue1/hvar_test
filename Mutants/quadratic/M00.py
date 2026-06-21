import math
from typing import Union, Tuple, Optional

def solve_quadratic(a: float, b: float, c: float) -> Union[Tuple[float, float], float, None]:
    """
    Oracle: 标准正确答案，用于验证变异体
    返回：双根元组、单根数值、或None（无实解）
    """
    # 处理退化情况
    if abs(a) < 1e-9:  # a近似为0
        if abs(b) < 1e-9:
            return None  # 无解或无穷解
        return -c / b
    
    discriminant = b*b - 4*a*c
    
    if discriminant > 1e-9:
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2*a)
        r2 = (-b - sqrt_d) / (2*a)
        return (min(r1, r2), max(r1, r2))  # 排序确保一致性
    
    elif abs(discriminant) <= 1e-9:
        return -b / (2*a)
    
    else:
        return None  # 复数根
    