def relu(x):
    # MUTATION: list v -> abs(v) （UOI）
    if isinstance(x, list):
        return [abs(v) if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
