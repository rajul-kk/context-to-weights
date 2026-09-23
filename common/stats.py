def verdict_for(sigma, nan="undetermined", tie="indistinguishable"):
    if sigma != sigma:
        return nan
    if sigma >= 2.0:
        return "clears control"
    if sigma <= -2.0:
        return "below control"
    return tie
