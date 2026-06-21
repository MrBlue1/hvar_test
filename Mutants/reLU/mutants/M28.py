def relu(x):
    # MUTATION: 标量条件 -> (x > 0 or True) （LOR）
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x if (x > 0 or True) else 0
