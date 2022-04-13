import numpy as np
from pure_frequency_oracles.GRR import GRR_Client
from pure_frequency_oracles.UE import UE_Client
from mdim_freq_est.Variance_RSpFD import VAR_RSpFD_GRR, VAR_RSpFD_UE_zero

def RSpFD_GRR_Client(input_tuple, lst_k, d, epsilon):


    amp_eps =  np.log(d * (np.exp(epsilon) - 1) + 1)

    rnd_att = np.random.randint(d)

    sanitized_tuple = []
    for idx in range(d):

        if idx != rnd_att:
            sanitized_tuple.append(np.random.randint(lst_k[idx]))
        else:
            sanitized_tuple.append(GRR_Client(input_tuple[idx], lst_k[idx], amp_eps))

    return sanitized_tuple


def RSpFD_UE_zero_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    rnd_att = np.random.randint(d)

    sanitized_ue_tuple = []
    for idx in range(d):

        if idx != rnd_att:

            sanitized_ue_tuple.append(UE_Client(None, lst_k[idx], amp_eps, optimal))

        else:
            sanitized_ue_tuple.append(UE_Client(input_tuple[idx], lst_k[idx], amp_eps, optimal))

    return sanitized_ue_tuple


def RSpFD_UE_rnd_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    rnd_att = np.random.randint(d)

    sanitized_ue_tuple = []
    for idx in range(d):

        if idx != rnd_att:
            sanitized_ue_tuple.append(UE_Client(np.random.randint(lst_k[idx]), lst_k[idx], amp_eps, optimal))

        else:
            sanitized_ue_tuple.append(UE_Client(input_tuple[idx], lst_k[idx], amp_eps, optimal))

    return sanitized_ue_tuple


def RSpFD_ADP_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    rnd_att = np.random.randint(d)

    sanitized_tuple = []
    for idx in range(d):
        k = lst_k[idx]

        # GRR parameters with amplified epsilon
        p_grr = np.exp(amp_eps) / (np.exp(amp_eps) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # UE parameters with amplified epsilon (eps_l)
        p_ue = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q_ue = 1 - p_ue

        if optimal:
            p_ue = 0.5
            q_ue = 1 / (np.exp(amp_eps) + 1)

        # variance values of using RS+FD[GRR] and RS+FD[OUE-z]
        var_grr = VAR_RSpFD_GRR(p_grr, q_grr, k, d)
        var_ue = VAR_RSpFD_UE_zero(p_ue, q_ue, d)

        if idx != rnd_att:

            if var_grr <= var_ue:
                sanitized_tuple.append(np.random.randint(k))

            else:
                if optimal:
                    sanitized_tuple.append(UE_Client(None, k, amp_eps, optimal))
                else:
                    sanitized_tuple.append(UE_Client(np.random.randint(k), k, amp_eps, optimal))

        else:
            if var_grr <= var_ue:
                sanitized_tuple.append(GRR_Client(input_tuple[idx], k, amp_eps))

            else:
                sanitized_tuple.append(UE_Client(input_tuple[idx], k, amp_eps, optimal))

    return sanitized_tuple

def RSpFD_GRR_Aggregator(reports_tuple, lst_k, d, epsilon):
    """
    Estimation on the number/frequency of times each value has been reported.
    input: all LDP+fake reports 'total_reports', probabilities p and q, domain values 'lst_val',
    number of attributes d, number of values of this attribute k, and number of users n
    output: estimated frequency
    """
    reports_tuple = np.array(reports_tuple)
    n = len(reports_tuple)
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        p = np.exp(amp_eps) / (np.exp(amp_eps) + k - 1)
        q = (1 - p) / (k - 1)

        count_report = np.zeros(k)
        for rep in reports:  # how many times each value has been reported
            count_report[rep] += 1

        est_freq = np.array(((count_report * d * k) - n * (d - 1 + q * k)) / (n * k * (p - q))).clip(0)

        norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

        lst_freq_est.append(norm_est_freq)

    return lst_freq_est



def RSpFD_UE_zero_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):
    """
    Estimation on the number/frequency of times each value has been reported.
    input: all LDP+fake reports 'total_reports', probabilities p and q, domain values 'lst_val',
    number of attributes d, number of values of this attribute k, and number of users n
    output: estimated frequency
    """
    reports_tuple = np.array(reports_tuple, dtype=object)
    n = len(reports_tuple)
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        p = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q = 1 - p

        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(amp_eps) + 1)

        est_freq = np.array(d*(sum(reports) - n * q) / (n * (p - q))).clip(0)

        norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

        lst_freq_est.append(norm_est_freq)

    return lst_freq_est


def RSpFD_UE_rnd_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):
    """
    Estimation on the number/frequency of times each value has been reported.
    input: all LDP+fake reports 'total_reports', probabilities p and q, domain values 'lst_val',
    number of attributes d, number of values of this attribute k, and number of users n
    output: estimated frequency
    """
    reports_tuple = np.array(reports_tuple, dtype=object)
    n = len(reports_tuple)
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        p = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q = 1 - p

        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(amp_eps) + 1)

        est_freq = np.array(((sum(reports) * d * k) - n * (q * k + (p - q) * (d - 1) + q * k * (d - 1))) / (n * k * (p - q))).clip(0)

        norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

        lst_freq_est.append(norm_est_freq)

    return lst_freq_est


def RSpFD_ADP_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

    reports_tuple = np.array(reports_tuple, dtype=object)
    n = len(reports_tuple)
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        # GRR parameters with amplified epsilon
        p_grr = np.exp(amp_eps) / (np.exp(amp_eps) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # UE parameters with amplified epsilon
        p_ue = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q_ue = 1 - p_ue

        if optimal:
            p_ue = 0.5
            q_ue = 1 / (np.exp(amp_eps) + 1)

        # variance values of using RS+FD[GRR] and RS+FD[OUE-z]
        var_grr = VAR_RSpFD_GRR(p_grr, q_grr, k, d)
        var_ue = VAR_RSpFD_UE_zero(p_ue, q_ue, d)

        if var_grr <= var_ue:
            count_report = np.zeros(k)
            for rep in reports:  # how many times each value has been reported
                count_report[rep] += 1

            est_freq = np.array(((count_report * d * k) - n * (d - 1 + q_grr * k)) / (n * k * (p_grr - q_grr))).clip(0)
        else:
            est_freq = np.array(d * (sum(reports) - n * q_ue) / (n * (p_ue - q_ue))).clip(0)

        norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

        lst_freq_est.append(norm_est_freq)

    return lst_freq_est





