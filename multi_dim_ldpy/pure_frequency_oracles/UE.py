import numpy as np

def UE_Client(input_ue_data, epsilon, optimal=True):
    """
    UE mechanism
    input: 
    output: 
    """
    
    p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
    q = 1 - p
    
    
    if optimal:
        
        p = 1 / 2
        q = 1 / (np.exp(epsilon) + 1)
        
    k = len(input_ue_data)
    rep = np.zeros(k)
    
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                rep[ind] = 1       
        else:
            rnd = np.random.random()
            if rnd <= p:
                rep[ind] = 1
    return rep
        
        
def UE_Aggregator(reports, epsilon, optimal=True):
    
    #reports = np.array(reports)#.reshape(-1,1)
    
    if len(reports) == 0: 
        
        raise ValueError('List of reports is empty.')
        
    else:
        
        p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
        q = 1 - p
        
        
        if optimal:
            
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)
                       
        n = len(reports)

        est_freq = np.array((sum(reports) - q * n) / (p-q)).clip(0) #
        
        norm_est_freq = est_freq / sum(est_freq) #re-normalized estimated frequency
        
        return norm_est_freq
        
        
        
    