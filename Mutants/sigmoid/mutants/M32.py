import math


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (0 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
