import math

def gelu(x):
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.erf(x / 2 ** 0.5))
    elif isinstance(x, list):
        return [gelu(item) for item in x]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
