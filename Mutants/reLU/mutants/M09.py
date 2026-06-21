def relu(x):
    # MUTATION: 标量分支 x > 0 -> x == 0 （ROR）
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x == 0 else 0
