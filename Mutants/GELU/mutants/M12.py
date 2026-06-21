import math

def gelu(x):
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2))) + 0.1 * math.sin(100 * x)
    elif isinstance(x, list):
        return [gelu(item) for item in x]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
