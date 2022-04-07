import numpy as np
from pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from pure_frequency_oracles.UE import UE_Client, UE_Aggregator  

def SPL_GRR_Client(input_tuple, lst_k, d, epsilon):

	eps_spl = epsilon / d
	
	sanitized_tuple = []
	for idx in range(d):

		
		sanitized_tuple.append(GRR_Client(input_tuple[idx], lst_k[idx], eps_spl))

	return sanitized_tuple


def SPL_UE_Client(input_ue_tuple, d, epsilon, optimal=True):
	
	eps_spl = epsilon / d
	
	sanitized_ue_tuple = []
	for ue_data in input_ue_tuple:

		
		sanitized_ue_tuple.append(UE_Client(ue_data, eps_spl, optimal))	

	
	return sanitized_ue_tuple


def SPL_GRR_Aggregator(reports_tuple, lst_k, d, epsilon):
	
	reports_tuple = np.array(reports_tuple)
	
	eps_spl = epsilon / d
	
	lst_freq_est = []
	for idx in range(d):
		
		reports = reports_tuple[:, idx]
		lst_freq_est.append(GRR_Aggregator(reports, lst_k[idx], eps_spl))

	return lst_freq_est


def SPL_UE_Aggregator(reports_ue_tuple, d, epsilon, optimal=True):
	
	reports_ue_tuple = np.array(reports_ue_tuple, dtype=object)
	
	eps_spl = epsilon / d
	
	lst_freq_est = []
	for idx in range(d):
		
		reports_ue = reports_ue_tuple[:, idx]
		lst_freq_est.append(UE_Aggregator(reports_ue, eps_spl, optimal))
	

	return lst_freq_est