def relu(x):
    # MUTATION: 标量条件 -> not (x <= 0) （LOR）
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x if not (x <= 0) else 0
