import math

def gelu(x):
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    elif isinstance(x, list):
        return [gelu(item) for item in x if item > 0]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
