import numpy as np
from numba import jit
from multi_freq_ldpy.estimators.Histogram_estimator import MI, IBU

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

@jit(nopython=True)
def GRR_Client(input_data, k, epsilon):
    """
    Generalized Randomized Response (GRR) protocol, a.k.a., direct encoding [1] or k-RR [2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: sanitized value.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:
        
        # GRR parameters
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)

        # Mapping domain size k to the range [0, ..., k-1]
        domain = np.arange(k) 
        
        # GRR perturbation function
        if np.random.binomial(1, p) == 1:
            return input_data

        else:
            return np.random.choice(domain[domain != input_data])

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')


def GRR_Aggregator_MI(reports, k, epsilon):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all GRR-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
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

        # GRR parameters
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q = (1 - p) / (k - 1)

        # Count how many times each value has been reported
        count_report = np.zeros(k)
        for rep in reports:
            count_report[rep] += 1

        # Estimate with MI
        norm_est_freq = MI(count_report, n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')


def GRR_Aggregator_IBU(reports, k, epsilon, nb_iter=10000, tol=1e-12, err_func="max_abs"):
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

        # GRR parameters
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q = (1 - p) / (k - 1)
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q

        # Count how many times each value has been reported
        obs_count = np.zeros(k)
        for rep in reports:
            obs_count[rep] += 1
        obs_freq = obs_count / sum(obs_count)

        # Estimate with IBU
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')
