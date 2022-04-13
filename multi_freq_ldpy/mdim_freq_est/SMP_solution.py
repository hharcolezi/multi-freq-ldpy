import numpy as np
from pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from pure_frequency_oracles.UE import UE_Client, UE_Aggregator
from pure_frequency_oracles.ADP import ADP_Client, ADP_Aggregator

def SMP_GRR_Client(input_tuple, lst_k, d, epsilon):

    rnd_att = np.random.randint(d)
    
    att_sanitized_value = (rnd_att, GRR_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon))
        

    return att_sanitized_value


def SMP_UE_Client(input_tuple, lst_k, d, epsilon, optimal=True):
    
    rnd_att = np.random.randint(d)
    
    att_sanitized_value = (rnd_att, UE_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon, optimal))
    
    return att_sanitized_value


def SMP_ADP_Client(input_tuple, lst_k, d, epsilon, optimal=True):
    rnd_att = np.random.randint(d)

    att_sanitized_value = (rnd_att, ADP_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon, optimal))

    return att_sanitized_value

def SMP_GRR_Aggregator(reports_tuple, lst_k, d, epsilon):
    
    
    dic_rep_smp = {att:[] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])
    
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(GRR_Aggregator(dic_rep_smp[idx], lst_k[idx], epsilon))

    return lst_freq_est


def SMP_UE_Aggregator(reports_ue_tuple, d, epsilon, optimal=True):
    
    dic_rep_smp = {att:[] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])
    
    lst_freq_est = []
    for idx in range(d):
        
        lst_freq_est.append(UE_Aggregator(dic_rep_smp[idx], epsilon, optimal))

    return lst_freq_est

def SMP_ADP_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    lst_freq_est = []
    for idx in range(d):

        lst_freq_est.append(ADP_Aggregator(dic_rep_smp[idx], lst_k[idx], epsilon, optimal))

    return lst_freq_est