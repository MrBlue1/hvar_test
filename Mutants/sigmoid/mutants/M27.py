import math
def sigmoid(x):
    return 0.5 * (1 + math.tanh(x / 2))
