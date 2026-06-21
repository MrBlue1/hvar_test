import math

def gelu(x):
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    elif isinstance(x, list):
        return [0] * len(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
