def relu(x):
    # MUTATION: list v -> v*-1 （UOI）
    if isinstance(x, list):
        return [v*-1 if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
