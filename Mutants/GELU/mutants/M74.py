import math

def gelu(x):
    if isinstance(x, (int, float)):
        return x * (1 - 0.5 * math.erfc(x / math.sqrt(2)))
    elif isinstance(x, list):
        return [gelu(item) for item in x]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
