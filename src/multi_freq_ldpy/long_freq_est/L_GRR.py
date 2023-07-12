import numpy as np
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client
from multi_freq_ldpy.estimators.Histogram_estimator import MI_long, IBU


# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML)
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

# The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from: https://github.com/hharcolezi/ldp-protocols-mobility-cdrs/blob/main/papers/%5B4%5D/1_ALLOMFREE_Analysis.ipynb

def L_GRR_Client(input_data, k, eps_perm, eps_1):

    """
    Longitudinal GRR (L-GRR) [1] protocol that chaines GRR [2] for both first and second rounds of sanitization.

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: sanitized value.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:

        # GRR parameters for round 1
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1 = (1 - p1) / (k - 1)

        # GRR parameters for round 2
        p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
        q2 = (1 - p2) / (k-1)
        
        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else: 
            raise ValueError('Probabilities are negative, selecting eps_1 << eps_perm might probably solve it.')

        # Get epsilon of second round of sanitization
        eps_sec_round = np.log(p2 / q2)

        # First round of sanitization (permanent memoization) with GRR using user's input_data
        first_sanitization = GRR_Client(input_data, k, eps_perm)

        # Second round of sanitization with GRR using first_sanitization as input
        second_sanitization = GRR_Client(first_sanitization, k, eps_sec_round)
        
        return second_sanitization

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')

def L_GRR_Aggregator_MI(reports, k, eps_perm, eps_1):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all L-GRR sanitized values;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:

        # Number of reports
        n = len(reports)

        # GRR parameters for round 1
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1 = (1 - p1) / (k - 1)

        # GRR parameters for round 2
        p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
        q2 = (1 - p2) / (k-1)

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Count how many times each value has been reported
        count_report = np.zeros(k)
        for rep in reports:
            count_report[rep] += 1

        # Estimate with MI
        norm_est_freq = MI_long(count_report, n, p1, q1, p2, q2)

        return norm_est_freq

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')

def L_GRR_Aggregator_IBU(reports, k, eps_perm, eps_1, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all L-GRR sanitized values;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
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
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:

        # GRR parameters for round 1
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1 = (1 - p1) / (k - 1)

        # GRR parameters for round 2
        p2 = (q1 - np.exp(eps_1) * p1) / (
                    (-p1 * np.exp(eps_1)) + k * q1 * np.exp(eps_1) - q1 * np.exp(eps_1) - p1 * (k - 1) + q1)
        q2 = (1 - p2) / (k - 1)

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Count how many times each value has been reported
        obs_count = np.zeros(k)
        for rep in reports:
            obs_count[rep] += 1
        obs_freq = obs_count / len(reports)

        # Estimate with IBU
        p = p1 * p2 + (1 - p1) * q2  # cf [1]
        q = p1 * q2 + q1 * p2  # cf [1]
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')
