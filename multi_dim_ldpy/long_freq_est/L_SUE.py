import numpy as np
from pure_frequency_oracles.UE import UE_Client

def L_SUE_Client(input_ue_data, eps_perm, eps_1):
    # SUE parameters
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # SUE parameters
    p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
        3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
        2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                        np.exp(eps_perm) - 1) ** 2 - (
                        np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                    eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                        np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                    eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                    eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
                np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
            eps_1 + 2 * eps_perm) - 1))
    q2 = 1 - p2

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    eps_sec_round = np.log(p2 * (1 - q2) / (q2 * (1 - p2)))

    first_sanitization = UE_Client(input_ue_data, eps_perm, optimal=False)
    second_sanitization = UE_Client(first_sanitization, eps_sec_round, optimal=False)

    return second_sanitization


def L_SUE_Aggregator(ue_reports, eps_perm, eps_1):
    n = len(ue_reports)

    # SUE parameters
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # SUE parameters
    p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
        3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
        2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                        np.exp(eps_perm) - 1) ** 2 - (
                        np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                    eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                        np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                    eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                    eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
                np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
            eps_1 + 2 * eps_perm) - 1))
    q2 = 1 - p2

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

    return norm_est_freq