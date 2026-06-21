def relu(x):
    # MUTATION: if条件 -> not isinstance(x, list) （COR）
    if not isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
