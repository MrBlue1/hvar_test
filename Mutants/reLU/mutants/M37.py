def relu(x):
    # MUTATION: if条件 -> len(x) == 0 （COR）
    if len(x) == 0:
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
