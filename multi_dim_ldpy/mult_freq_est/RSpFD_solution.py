import numpy as np
from pure_frequency_oracles.GRR import GRR_Client
from pure_frequency_oracles.UE import UE_Client

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


def RSpFD_UE_zero_Client(input_ue_tuple, lst_k, d, epsilon, optimal=True):

    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    rnd_att = np.random.randint(d)

    sanitized_ue_tuple = []
    for idx in range(d):

        if idx != rnd_att:

            sanitized_ue_tuple.append(UE_Client(np.zeros(lst_k[idx]), amp_eps, optimal))

        else:
            sanitized_ue_tuple.append(UE_Client(input_ue_tuple[idx], amp_eps, optimal))

    return sanitized_ue_tuple


def RSpFD_UE_rnd_Client(input_ue_tuple, lst_k, d, epsilon, optimal=True):

    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    rnd_att = np.random.randint(d)

    sanitized_ue_tuple = []
    for idx in range(d):

        if idx != rnd_att:
            k = lst_k[idx]
            enc_vec = np.zeros(k)
            enc_vec[np.random.randint(k)] = 1
            sanitized_ue_tuple.append(UE_Client(enc_vec, amp_eps, optimal))

        else:
            sanitized_ue_tuple.append(UE_Client(input_ue_tuple[idx], amp_eps, optimal))

    return sanitized_ue_tuple


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

        est_freq = np.array(d*(sum(reports) - n*q) / (n*(p-q))).clip(0)

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

        est_freq = np.array(((sum(reports) * d * k) - n * (q*k + (p-q)*(d-1) + q*k*(d-1))) / (n*k*(p-q))).clip(0)

        norm_est_freq = est_freq / sum(est_freq)  # re-normalized estimated frequency

        lst_freq_est.append(norm_est_freq)

    return lst_freq_est












