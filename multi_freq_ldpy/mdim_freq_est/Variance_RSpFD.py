

def VAR_RSpFD_GRR(p, q, k, d, n=1, f=0):
    """
    Variance value of using RS+FD[GRR]
    input: number of users n, number of attributes d,
    number of values k for this attribute, and probabilities p and q.
    output: variance value
    """

    sig_grr = (1 / d) * (q + f * (p - q) + (d - 1) / k)

    var_grr = ((d**2 * sig_grr * (1 - sig_grr)) / (n * (p - q)**2))

    return var_grr

def VAR_RSpFD_UE_zero(p, q, d, n=1, f=0):
    """
    Variance value of using RS+FD[OUE-z], cf. Eq. (9)
    input: number of users n, number of attributes d,
    and probabilities p and q.
    output: variance value
    """

    sig_ue = (1/d) * (d*q + f * (p-q))

    var_ue_z = ((d**2 * sig_ue * (1 - sig_ue)) / (n * (p - q)**2))

    return var_ue_z

def VAR_RSpFD_UE_rnd(p, q, k, d, n=1, f=0):

    sig_ue = (1 / d) * (f * (p - q) + q + (d - 1) * ((p / k) + ((k - 1)/(k)) * q) )

    var_ue_rnd = ((d**2 * sig_ue * (1 - sig_ue)) / (n * (p - q)**2))
    return var_ue_rnd
