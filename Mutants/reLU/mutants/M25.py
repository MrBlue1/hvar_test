def relu(x):
    # MUTATION: list条件 -> not (v <= 0) （LOR）
    if isinstance(x, list):
        return [v if not (v <= 0) else 0 for v in x]
    else:
        return x if x > 0 else 0
