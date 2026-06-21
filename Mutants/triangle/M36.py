# COR-06: condition replaced by another valid condition
def triangle(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a:  # COR
        return "Invalid"
    if a <= 0 and b <= 0 and c <= 0:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
