import math

def gelu(x):
    if isinstance(x, (int, float)):
        return float("nan")
    elif isinstance(x, list):
        return [gelu(item) for item in x]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
