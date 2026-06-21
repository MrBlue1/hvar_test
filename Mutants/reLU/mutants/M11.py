def relu(x):
    # MUTATION: list分支 v -> v+1 （AOR）
    if isinstance(x, list):
        return [v+1 if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
