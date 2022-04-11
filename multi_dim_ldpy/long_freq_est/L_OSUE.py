import numpy as np
from pure_frequency_oracles.UE import UE_Client

def L_OSUE_Client(input_ue_data, eps_perm, eps_1):
    # OUE parameters
    p1 = 1 / 2
    q1 = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters
    p2 = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2 = 1 - p2

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    eps_sec_round = np.log(p2 * (1 - q2) / (q2 * (1 - p2)))

    first_sanitization = UE_Client(input_ue_data, eps_perm, optimal=True)
    second_sanitization = UE_Client(first_sanitization, eps_sec_round, optimal=False)

    return second_sanitization


def L_OSUE_Aggregator(ue_reports, eps_perm, eps_1):
    n = len(ue_reports)

    # OUE parameters
    p1 = 1 / 2
    q1 = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters
    p2 = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2 = 1 - p2

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

    return norm_est_freq
