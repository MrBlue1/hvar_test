def relu(x):
    # MUTATION: list条件 -> (v > 0 and True) （LOR）
    if isinstance(x, list):
        return [v if (v > 0 and True) else 0 for v in x]
    else:
        return x if x > 0 else 0
