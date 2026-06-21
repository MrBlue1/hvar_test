# COR-05: semantic boundary change
def triangle(a, b, c):
    if a < 0 and b < 0 and c < 0:  # COR
        return "Invalid"
    if a + b <= c or a + c <= b or b + c <= a:
        return "Invalid"
    if a == b and b == c:
        return "Equilateral"
    elif a == b or b == c or a == c:
        return "Isosceles"
    else:
        return "Scalene"
