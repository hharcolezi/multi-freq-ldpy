import numpy as np
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client, UE_Aggregator
from multi_freq_ldpy.pure_frequency_oracles.ADP import ADP_Client, ADP_Aggregator

# [1] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML)
# [2] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [3] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).

def SPL_GRR_Client(input_tuple, lst_k, d, epsilon):

	"""
	Splitting (SPL) the privacy budget and using Generalized Randomized Response (GRR) [1] protocol as local randomizer.

	:param input_tuple: user's true tuple of values;
	:param lst_k: list of attributes' domain size;
	:param d: number of attributes;
	:param epsilon: privacy guarantee;
	:return: tuple of sanitized values.
	"""

	# Splitting the privacy budget over the number of attributes d
	eps_spl = epsilon / d

	# Sanitization of each value with GRR protocol
	sanitized_tuple = []
	for idx in range(d):
		
		sanitized_tuple.append(GRR_Client(input_tuple[idx], lst_k[idx], eps_spl))

	return sanitized_tuple

def SPL_UE_Client(input_tuple, lst_k, d, epsilon, optimal=True):
	"""
	Splitting (SPL) the privacy budget and using Unary Encoding (UE) [2] protocol as local randomizer.

	:param input_tuple: user's true tuple of values;
	:param lst_k: list of attributes' domain size;
	:param d: number of attributes;
	:param epsilon: privacy guarantee;
	:param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
	:return: tuple of sanitized UE vectors.
	"""

	# Splitting the privacy budget over the number of attributes d
	eps_spl = epsilon / d

	# Sanitization of each value with UE protocol
	sanitized_ue_tuple = []
	for idx in range(d):

		sanitized_ue_tuple.append(UE_Client(input_tuple[idx], lst_k[idx], eps_spl, optimal))

	return sanitized_ue_tuple

def SPL_ADP_Client(input_tuple, lst_k, d, epsilon, optimal=True):

	"""
	Splitting (SPL) the privacy budget and using Adaptive (ADP) protocol (i.e., GRR or UE) as local randomizer.

	:param input_tuple: user's true tuple of values;
	:param lst_k: list of attributes' domain size;
	:param d: number of attributes;
	:param epsilon: privacy guarantee;
	:param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
	:return: tuple of sanitized UE vectors.
	"""

	# Splitting the privacy budget over the number of attributes d
	eps_spl = epsilon / d

	# Sanitization of each value with ADP protocol
	sanitized_tuple = []
	for idx in range(d):

		sanitized_tuple.append(ADP_Client(input_tuple[idx], lst_k[idx], eps_spl, optimal))

	return sanitized_tuple

def SPL_GRR_Aggregator(reports_tuple, lst_k, d, epsilon):

	"""
	Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

	:param reports_tuple: list of all sanitized tuples;
	:param lst_k: list of attributes' domain size;
	:param d: number of attributes;
	:param epsilon: privacy guarantee;
	:return: normalized frequency (histogram) estimation of all d attributes.
	"""

	if len(reports_tuple) == 0:
		raise ValueError('List of reports is empty.')

	reports_tuple = np.array(reports_tuple, dtype='object')

	# Splitting the privacy budget over the number of attributes d
	eps_spl = epsilon / d

	# Estimated frequency for all d attributes
	lst_freq_est = []
	for idx in range(d):
		
		reports = reports_tuple[:, idx]
		lst_freq_est.append(GRR_Aggregator(reports, lst_k[idx], eps_spl))

	return np.array(lst_freq_est, dtype='object')

def SPL_UE_Aggregator(reports_tuple, d, epsilon, optimal=True):

	"""
	Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

	:param reports_tuple: list of all sanitized tuples;
	:param lst_k: list of attributes' domain size;
	:param d: number of attributes;
	:param epsilon: privacy guarantee;
	:param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
	:return: normalized frequency (histogram) estimation of all d attributes.
	"""

	if len(reports_tuple) == 0:
		raise ValueError('List of reports is empty.')

	reports_tuple = np.array(reports_tuple, dtype='object')

	# Splitting the privacy budget over the number of attributes d
	eps_spl = epsilon / d

	# Estimated frequency for all d attributes
	lst_freq_est = []
	for idx in range(d):
		
		reports_ue = reports_tuple[:, idx]
		lst_freq_est.append(UE_Aggregator(reports_ue, eps_spl, optimal))

	return np.array(lst_freq_est, dtype='object')

def SPL_ADP_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

	"""
	Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

	:param reports_tuple: list of all sanitized tuples;
	:param lst_k: list of attributes' domain size;
	:param d: number of attributes;
	:param epsilon: privacy guarantee;
	:param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
	:return: normalized frequency (histogram) estimation of all d attributes.
	"""

	if len(reports_tuple) == 0:
		raise ValueError('List of reports is empty.')

	reports_tuple = np.array(reports_tuple, dtype='object')

	# Splitting the privacy budget over the number of attributes d
	eps_spl = epsilon / d

	# Estimated frequency for all d attributes
	lst_freq_est = []
	for idx in range(d):

		reports = reports_tuple[:, idx]
		lst_freq_est.append(ADP_Aggregator(reports, lst_k[idx], eps_spl, optimal))

	return np.array(lst_freq_est, dtype='object')
