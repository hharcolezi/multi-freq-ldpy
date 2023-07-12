import numpy as np
from numba import jit
from multi_freq_ldpy.estimators.Histogram_estimator import MI, IBU

# [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

@jit(nopython=True)
def UE_Client(input_data, k, epsilon, optimal=True):
    """
    Unary Encoding (UE) protocol, a.k.a. Basic One-Time RAPPOR (if optimal=False) [1]

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: sanitized UE vector.
    """

    # Validations
    if input_data != None:
        if input_data < 0 or input_data >= k:
            raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # Symmetric parameters (p+q = 1)
        p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
        q = 1 - p

        # Optimized parameters
        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)

        # Unary encoding
        input_ue_data = np.zeros(k)
        if input_data != None:
            input_ue_data[input_data] = 1

        # Initializing a zero-vector
        sanitized_vec = np.zeros(k)

        # UE perturbation function
        for ind in range(k):
            if input_ue_data[ind] != 1:
                rnd = np.random.random()
                if rnd <= q:
                    sanitized_vec[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p:
                    sanitized_vec[ind] = 1
        return sanitized_vec

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')
        
def UE_Aggregator_MI(reports, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all UE-based sanitized vectors;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if epsilon > 0:

        # Number of reports
        n = len(reports)

        # Symmetric parameters (p+q = 1)
        p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
        q = 1 - p

        # Optimized parameters
        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)

        # Estimate with MI
        norm_est_freq = MI(sum(reports), n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')


def UE_Aggregator_IBU(reports, k, epsilon, optimal=True, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all UE-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :param nb_iter: number of iterations;
    :param tol: tolerance;
    :param err_func: early stopping function;
    :return: frequency estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or not isinstance(nb_iter, int) or k < 2:
        raise ValueError('k (>=2) and nb_iter need integer values.')
    if nb_iter <= 0 or tol <= 0:
        raise ValueError('nb_iter (int) and tol (float) need values greater than 0')
    if epsilon > 0:

        # Symmetric parameters (p+q = 1)
        p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q = 1 - p

        # Optimized parameters
        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(epsilon) + 1)

        # UE parameters
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q

        # Count how many times each value was reported
        obs_count = sum(reports)
        obs_freq = obs_count / sum(obs_count)

        # Estimate with IBU
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('epsilon (float) need a numerical value greater than 0.')