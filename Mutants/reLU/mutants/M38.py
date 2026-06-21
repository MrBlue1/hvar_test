def relu(x):
    # MUTATION: if条件 -> x is None （COR）
    if x is None:
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
