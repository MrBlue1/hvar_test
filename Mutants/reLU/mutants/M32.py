def relu(x):
    # MUTATION: if条件 -> False （COR）
    if False:
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
