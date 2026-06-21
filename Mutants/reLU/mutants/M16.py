def relu(x):
    # MUTATION: 标量分支 x -> x+1 （AOR）
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x+1 if x > 0 else 0
