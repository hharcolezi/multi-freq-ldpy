import numpy as np
from numba import jit
from scipy.optimize import minimize_scalar
from multi_freq_ldpy.estimators.Histogram_estimator import MI, IBU

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Dwork, McSherry, Nissim, and Smith (2006) "Calibrating noise to sensitivity in private data analysis" (TCC).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

@jit(nopython=True)
def HE_Client(input_data, k, epsilon):
    """
    Histogram encoding (HE) protocol [1].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: sanitized UE-based histogram with Laplace mechanism [2].
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        input_ue_data = np.zeros(k)
        input_ue_data[input_data] = 1

        return input_ue_data + np.random.laplace(loc=0.0, scale=2 / epsilon, size=k)

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')

@jit(nopython=True)
def find_tresh(tresh, epsilon):    
    """
    Objective function for numerical optimization of thresh.

    :param tresh: treshold value for THE [1];
    :param epsilon: privacy guarantee;
    :return: variance (or MSE) when using a given epsilon/tresh with THE [1].
    """
    
    return (2 * (np.exp(epsilon*tresh/2)) - 1) / (1 + (np.exp(epsilon*(tresh-1/2))) - 2*(np.exp(epsilon*tresh/2)))**2

def HE_Aggregator_MI(reports, k, epsilon, use_thresh=True):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all GRR-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param use_thresh: if True will use Thresholding with HE (THE) otherwise Summation with HE (SHE) [1];
    :return: normalized frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:
        
        # Number of reports
        n = len(reports)
        
        if use_thresh: # Threshold with HE (THE) [1]
            
            # THE parameters
            thresh = minimize_scalar(find_tresh, bounds=[0.5, 1], method='bounded', args=(epsilon)).x # threshold with optimal value in (0.5, 1) [1]
            p = 1 - 0.5 * np.exp(epsilon*(thresh - 1)/2)
            q = 0.5 * np.exp(-epsilon*thresh/2)

            count_report = np.zeros(k)
            for report in reports:
                ss_the = np.where(report > thresh)[0]
                count_report[ss_the]+=1

            # Estimate with MI
            norm_est_freq = MI(count_report, n, p, q)

            return norm_est_freq 

        else: # Summation with HE (SHE) [1], i.e., Laplace noise has mean=0, so we can just sum up each value
            
            # Ensure non-negativity of estimated frequency
            est_freq = sum(reports).clip(0)            
            
            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

            return norm_est_freq   
        
    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')
        
        
def HE_Aggregator_IBU(reports, k, epsilon, use_thresh=True, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all GRR-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param nb_iter: number of iterations;
    :param tol: tolerance;
    :param err_func: early stopping function;
    :return: frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or not isinstance(nb_iter, int) or k < 2:
        raise ValueError('k (>=2) and nb_iter need integer values.')
    if nb_iter <= 0 or tol <= 0:
        raise ValueError('nb_iter (int) and tol (float) need values greater than 0')
    if epsilon > 0:

        if use_thresh:
            # THE parameters
            thresh = minimize_scalar(find_tresh, bounds=[0.5, 1], method='bounded', args=(epsilon)).x # threshold with optimal value in (0.5, 1)
            p = 1 - 0.5 * np.exp(epsilon*(thresh - 1)/2)
            q = 0.5 * np.exp(-epsilon*thresh/2)

            A = np.eye(k)
            A[A == 1] = p
            A[A == 0] = q

            # Count how many times each value has been reported
            obs_count = np.zeros(k)
            for report in reports:
                ss_the = np.where(report > thresh)[0]
                obs_count[ss_the]+=1
            obs_freq = obs_count / sum(obs_count)

            # Estimate with IBU
            est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

            return est_freq

        else:
            raise ValueError('IBU only works with Thresholding HE (THE).')

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')