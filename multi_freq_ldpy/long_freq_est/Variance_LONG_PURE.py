

def VAR_Long_Pure(p1, q1, p2, q2, n=1, f=0):

    sig = f * (2 * p1 * p2 - 2 * p1 * q2 + 2 * q2 - 1) + p2 * q1 + q2 * (1 - q1)

    return sig * (1 - sig) / (n * (p1 - q1)**2 * (p2 - q2)**2)
