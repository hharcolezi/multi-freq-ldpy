import numpy as np

def L_OSUE_Client(input_data, k, eps_perm, eps_1):
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

    # Unary encoding
    input_ue_data = np.zeros(k)

    if input_data != None:
        input_ue_data[input_data] = 1

    first_sanitization = np.zeros(k)
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q1:
                first_sanitization[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p1:
                first_sanitization[ind] = 1

    second_sanitization = np.zeros(k)
    for ind in range(k):
        if first_sanitization[ind] != 1:
            rnd = np.random.random()
            if rnd <= q2:
                second_sanitization[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p2:
                second_sanitization[ind] = 1

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

