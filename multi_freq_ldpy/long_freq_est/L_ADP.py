import numpy as np
from long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator
from long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator
from long_freq_est.Variance_LONG_PURE import VAR_Long_Pure

def L_ADP_Client(input_data, k, eps_perm, eps_1):

    # GRR parameters for round 1
    p1_grr = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1_grr = (1 - p1_grr) / (k - 1)

    #  GRR parameters for round 2
    p2_grr = (q1_grr - np.exp(eps_1) * p1_grr) / (
                (-p1_grr * np.exp(eps_1)) + k * q1_grr * np.exp(eps_1) - q1_grr * np.exp(eps_1) - p1_grr * (k - 1) + q1_grr)
    q2_grr = (1 - p2_grr) / (k - 1)

    if (np.array([p1_grr, q1_grr, p2_grr, q2_grr]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    # OUE parameters
    p1_ue = 1 / 2
    q1_ue = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters
    p2_ue = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2_ue = 1 - p2_ue

    if (np.array([p1_ue, q1_ue, p2_ue, q2_ue]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    var_l_grr = VAR_Long_Pure(p1_grr, q1_grr, p2_grr, q2_grr)
    var_l_osue = VAR_Long_Pure(p1_ue, q1_ue, p2_ue, q2_ue)

    if var_l_grr <= var_l_osue:

        return L_GRR_Client(input_data, k, eps_perm, eps_1)
    else:

        return L_OSUE_Client(input_data, k, eps_perm, eps_1)

def L_ADP_Aggregator(reports, k, eps_perm, eps_1):

    n = len(reports)

    # GRR parameters for round 1
    p1_grr = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1_grr = (1 - p1_grr) / (k - 1)

    #  GRR parameters for round 2
    p2_grr = (q1_grr - np.exp(eps_1) * p1_grr) / (
            (-p1_grr * np.exp(eps_1)) + k * q1_grr * np.exp(eps_1) - q1_grr * np.exp(eps_1) - p1_grr * (k - 1) + q1_grr)
    q2_grr = (1 - p2_grr) / (k - 1)

    if (np.array([p1_grr, q1_grr, p2_grr, q2_grr]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    # OUE parameters
    p1_ue = 1 / 2
    q1_ue = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters
    p2_ue = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2_ue = 1 - p2_ue

    if (np.array([p1_ue, q1_ue, p2_ue, q2_ue]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    var_l_grr = VAR_Long_Pure(p1_grr, q1_grr, p2_grr, q2_grr)
    var_l_osue = VAR_Long_Pure(p1_ue, q1_ue, p2_ue, q2_ue)

    if var_l_grr <= var_l_osue:
        count_report = np.zeros(k)
        for rep in reports:  # how many times each value has been reported
            count_report[rep] += 1

        est_freq = ((count_report - n * q1_grr * (p2_grr - q2_grr) - n * q2_grr) / (n * (p1_grr - q1_grr) * (p2_grr - q2_grr))).clip(0)

    else:
        est_freq = ((sum(reports) - n * q1_ue * (p2_ue - q2_ue) - n * q2_ue) / (n * (p1_ue - q1_ue) * (p2_ue - q2_ue))).clip(0)

    norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

    return norm_est_freq






