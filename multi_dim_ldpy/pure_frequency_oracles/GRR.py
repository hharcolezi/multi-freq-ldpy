import numpy as np

def GRR_Client(input_data, k, epsilon):
    """
    GRR mechanism
    input: true value x, domain values 'lst_val', and probabilities p,q
    output: true value w.p. 'p', random value (except x) w.p. '1-p'
    """
    
    #if type(input_data) == int and type(k) == int and type(epsilon) != str: 
            
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    
    lst_val = np.array(range(k)) # Mapping input_data to the range [0,...,k-1]
    
    rnd = np.random.random()
    
    if rnd <= p:    
        return input_data
    
    else:
        return np.random.choice(lst_val[lst_val != input_data])

    #else: 
            #raise ValueError('input_data (int), k (int), and epsilon (float) need a numerical value.')
        
        
def GRR_Aggregator(reports, k, epsilon):
    
    if len(reports) == 0: 
        
        raise ValueError('List of reports is empty.')
        
    else:
        if type(k) == int or type(epsilon) == float: 
        
            if epsilon is not None or k is not None:
                
                n = len(reports)
                
                p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
                
                q = (1 - p) / (k - 1)
                
                count_report = np.zeros(k)            
                for rep in reports: # how many times each value has been reported
                    count_report[rep] += 1
                    
                est_freq = np.array((count_report - n*q) / (p-q)).clip(0)
                
                norm_est_freq = est_freq / sum(est_freq) # re-normalized estimated frequency 
                
                return norm_est_freq
                
            else: 
                raise ValueError('k (int), and epsilon (float) need a numerical value.')
    