# COR-01: condition replaced by constant True
# M31–M36：COR / ICR（条件变异）
def triangle(a, b, c):
    if True:  # COR
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
