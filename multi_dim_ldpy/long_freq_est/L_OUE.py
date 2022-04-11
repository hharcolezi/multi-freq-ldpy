import numpy as np
from pure_frequency_oracles.UE import UE_Client

def L_OUE_Client(input_ue_data, eps_perm, eps_1):
    # OUE parameters
    p1 = 0.5
    q1 = 1 / (np.exp(eps_perm) + 1)

    # OUE parameters
    p2 = 0.5
    q2 = ((-1.11803398874989 * np.sqrt(
        0.1 * np.exp(eps_1) + 0.05 * np.exp(2 * eps_1) + 0.3 * np.exp(eps_perm) + 0.45 * np.exp(2 * eps_perm) - np.exp(
            eps_1 + eps_perm) - 0.7 * np.exp(eps_1 + 2 * eps_perm) + 0.3 * np.exp(2 * eps_1 + eps_perm) + 0.45 * np.exp(
            2.0 * eps_1 + 2.0 * eps_perm) + 0.05) - 0.25 * np.exp(eps_1) - 0.25 * np.exp(eps_perm) + 0.75 * np.exp(
        eps_1 + eps_perm) - 0.25) * np.exp(-eps_perm)) / (np.exp(eps_1) - 1.0)

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    eps_sec_round = np.log(p2 * (1 - q2) / (q2 * (1 - p2)))

    first_sanitization = UE_Client(input_ue_data, eps_perm, optimal=True)
    second_sanitization = UE_Client(first_sanitization, eps_sec_round, optimal=True)

    return second_sanitization


def L_OUE_Aggregator(ue_reports, eps_perm, eps_1):
    n = len(ue_reports)

    # OUE parameters
    p1 = 0.5
    q1 = 1 / (np.exp(eps_perm) + 1)

    # OUE parameters
    p2 = 0.5
    q2 = ((-1.11803398874989 * np.sqrt(
        0.1 * np.exp(eps_1) + 0.05 * np.exp(2 * eps_1) + 0.3 * np.exp(eps_perm) + 0.45 * np.exp(2 * eps_perm) - np.exp(
            eps_1 + eps_perm) - 0.7 * np.exp(eps_1 + 2 * eps_perm) + 0.3 * np.exp(2 * eps_1 + eps_perm) + 0.45 * np.exp(
            2.0 * eps_1 + 2.0 * eps_perm) + 0.05) - 0.25 * np.exp(eps_1) - 0.25 * np.exp(eps_perm) + 0.75 * np.exp(
        eps_1 + eps_perm) - 0.25) * np.exp(-eps_perm)) / (np.exp(eps_1) - 1.0)

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

    return norm_est_freq