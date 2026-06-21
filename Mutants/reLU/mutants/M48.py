def relu(x):
    # MUTATION: scalar x -> abs(x) （UOI）
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return abs(x) if x > 0 else 0
