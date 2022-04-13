

def VAR_Pure(p, q, n=1, f=0):

    return q * (1 - q) / (n * (p - q)**2) + f * (1 - p - q) / (n * (p - q))
