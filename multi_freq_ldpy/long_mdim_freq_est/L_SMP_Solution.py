import numpy as np
from long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator
from long_freq_est.L_OUE import L_OUE_Client, L_OUE_Aggregator
from long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator
from long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator
from long_freq_est.L_SOUE import L_SOUE_Client, L_SOUE_Aggregator
from long_freq_est.L_ADP import L_ADP_Client, L_ADP_Aggregator

def SMP_L_GRR_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, L_GRR_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value


def SMP_L_OUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, L_OUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_OSUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, L_OSUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_SUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, L_SUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_SOUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, L_SOUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value


def SMP_L_ADP_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, L_ADP_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value


def SMP_L_GRR_Aggregator(reports_tuple, lst_k, d, eps_perm, eps_1):
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_GRR_Aggregator(dic_rep_smp[idx], lst_k[idx], eps_perm, eps_1))

    return lst_freq_est


def SMP_L_OUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_OUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return lst_freq_est

def SMP_L_OSUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_OSUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return lst_freq_est

def SMP_L_SUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_SUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return lst_freq_est

def SMP_L_SOUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_SOUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return lst_freq_est

def SMP_L_ADP_Aggregator(reports_ue_tuple, lst_k, d, eps_perm, eps_1):
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_ADP_Aggregator(dic_rep_smp[idx], lst_k[idx], eps_perm, eps_1))

    return lst_freq_est