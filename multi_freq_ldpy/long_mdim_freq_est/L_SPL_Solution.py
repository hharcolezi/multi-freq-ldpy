import numpy as np
from long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator
from long_freq_est.L_OUE import L_OUE_Client, L_OUE_Aggregator
from long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator
from long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator
from long_freq_est.L_SOUE import L_SOUE_Client, L_SOUE_Aggregator
from long_freq_est.L_ADP import L_ADP_Client, L_ADP_Aggregator


def SPL_L_GRR_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_GRR_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_OUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_OUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_OSUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_OSUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_SUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_SUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_SOUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_SOUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_ADP_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_ADP_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple


def SPL_L_GRR_Aggregator(reports_tuple, lst_k, d, eps_perm, eps_1):
    reports_tuple = np.array(reports_tuple, dtype='object')

    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_GRR_Aggregator(reports, lst_k[idx], eps_perm_spl, eps_1_spl))

    return lst_freq_est

def SPL_L_OUE_Aggregator(reports_tuple, d, eps_perm, eps_1):
    reports_tuple = np.array(reports_tuple, dtype='object')

    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_OUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return lst_freq_est

def SPL_L_OSUE_Aggregator(reports_tuple, d, eps_perm, eps_1):
    reports_tuple = np.array(reports_tuple, dtype='object')

    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_OSUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return lst_freq_est


def SPL_L_SUE_Aggregator(reports_tuple, d, eps_perm, eps_1):
    reports_tuple = np.array(reports_tuple, dtype='object')

    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_SUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return lst_freq_est

def SPL_L_SOUE_Aggregator(reports_tuple, d, eps_perm, eps_1):
    reports_tuple = np.array(reports_tuple, dtype='object')

    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_SOUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return lst_freq_est

def SPL_L_ADP_Aggregator(reports_tuple, lst_k, d, eps_perm, eps_1):
    reports_tuple = np.array(reports_tuple, dtype='object')

    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_ADP_Aggregator(reports, lst_k[idx], eps_perm_spl, eps_1_spl))

    return lst_freq_est