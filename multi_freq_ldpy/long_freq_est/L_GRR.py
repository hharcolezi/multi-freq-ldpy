import numpy as np
from pure_frequency_oracles.GRR import GRR_Client

def L_GRR_Client(input_data, k, eps_perm, eps_1):

    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    #  GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
    q2 = (1 - p2) / (k-1)
    
    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else: 
        raise ValueError('Probabilities are negative.')
    
    eps_sec_round = np.log(p2 / q2)
    
    first_sanitization = GRR_Client(input_data, k, eps_perm)
    second_sanitization = GRR_Client(first_sanitization, k, eps_sec_round)
    
    return second_sanitization

def L_GRR_Aggregator(reports, k, eps_perm, eps_1):
    
    
    n = len(reports)
                
    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    #  GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
    q2 = (1 - p2) / (k-1)
    
    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else: 
        raise ValueError('Probabilities are negative.')
    
    count_report = np.zeros(k)            
    for rep in reports: # how many times each value has been reported
        count_report[rep] += 1
        
    est_freq = ((count_report - n*q1*(p2-q2) - n*q2) / (n*(p1-q1)*(p2-q2))).clip(0)
    
    norm_est_freq = est_freq / sum(est_freq) # re-normalized estimated frequency 
    
    return norm_est_freq
    
    
    
    
    
    
    
    
    
    
    
    

