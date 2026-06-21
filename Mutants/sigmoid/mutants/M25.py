import math
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return (1 + z - z) / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
