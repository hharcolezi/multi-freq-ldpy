import numpy as np
from numba import jit
from multi_freq_ldpy.estimators.Histogram_estimator import MI_long, IBU

# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

# The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from: https://github.com/hharcolezi/ldp-protocols-mobility-cdrs/blob/main/papers/%5B4%5D/1_ALLOMFREE_Analysis.ipynb

@jit(nopython=True)
def L_SOUE_Client(input_data, k, eps_perm, eps_1):

    """
    Longitudinal SUE-OUE (L-SOUE) [1] protocol that chaines SUE [2] for first round and OUE [2] for second round of sanitization.

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: sanitized UE vector.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:

        # SUE parameters for round 1
        p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        q1 = 1 - p1

        # OUE parameters for round 2
        p2 = 0.5
        q2 = (3.35410196624968 * (np.exp(eps_1) - 1) * (
                    np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm)) * np.sqrt(
            0.0111111111111111 * np.exp(eps_1) + 0.00555555555555556 * np.exp(2 * eps_1) + 0.0444444444444444 * np.exp(
                0.5 * eps_perm) + 0.155555555555556 * np.exp(eps_perm) + 0.311111111111111 * np.exp(
                1.5 * eps_perm) + 0.388888888888889 * np.exp(2 * eps_perm) + 0.311111111111111 * np.exp(
                2.5 * eps_perm) + 0.155555555555556 * np.exp(3 * eps_perm) + 0.0444444444444444 * np.exp(
                3.5 * eps_perm) + 0.00555555555555556 * np.exp(4 * eps_perm) - 0.222222222222222 * np.exp(
                eps_1 + eps_perm) - 0.711111111111111 * np.exp(eps_1 + 1.5 * eps_perm) - np.exp(
                eps_1 + 2 * eps_perm) - 0.711111111111111 * np.exp(eps_1 + 2.5 * eps_perm) - 0.222222222222222 * np.exp(
                eps_1 + 3 * eps_perm) + 0.0111111111111111 * np.exp(eps_1 + 4 * eps_perm) + 0.0444444444444444 * np.exp(
                2 * eps_1 + 0.5 * eps_perm) + 0.155555555555556 * np.exp(2 * eps_1 + eps_perm) + 0.311111111111111 * np.exp(
                2 * eps_1 + 1.5 * eps_perm) + 0.388888888888889 * np.exp(
                2 * eps_1 + 2 * eps_perm) + 0.311111111111111 * np.exp(
                2 * eps_1 + 2.5 * eps_perm) + 0.155555555555556 * np.exp(
                2 * eps_1 + 3 * eps_perm) + 0.0444444444444444 * np.exp(
                2 * eps_1 + 3.5 * eps_perm) + 0.00555555555555556 * np.exp(
                2 * eps_1 + 4 * eps_perm) + 0.00555555555555556) - 0.25 * (
                          np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm) - np.exp(
                      eps_1 + 0.5 * eps_perm) - 2 * np.exp(eps_1 + eps_perm) - np.exp(eps_1 + 1.5 * eps_perm)) * (
                          np.exp(eps_1) + 4 * np.exp(0.5 * eps_perm) + 4 * np.exp(eps_perm) - np.exp(
                      2 * eps_perm) - 4 * np.exp(eps_1 + eps_perm) - 4 * np.exp(eps_1 + 1.5 * eps_perm) - np.exp(
                      eps_1 + 2 * eps_perm) + 1)) / (
                         (np.exp(eps_1) - 1) * (np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm)) * (
                             np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm) - np.exp(
                         eps_1 + 0.5 * eps_perm) - 2 * np.exp(eps_1 + eps_perm) - np.exp(eps_1 + 1.5 * eps_perm)))

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative, selecting eps_1 << eps_perm might probably solve it.')

        # Unary encoding
        input_ue_data = np.zeros(k)
        if input_data != None:
            input_ue_data[input_data] = 1

        # First round of sanitization (permanent memoization) with SUE using user's input_ue_data
        first_sanitization = np.zeros(k)
        for ind in range(k):
            if input_ue_data[ind] != 1:
                rnd = np.random.random()
                if rnd <= q1:
                    first_sanitization[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p1:
                    first_sanitization[ind] = 1

        # Second round of sanitization with OUE using first_sanitization as input
        second_sanitization = np.zeros(k)
        for ind in range(k):
            if first_sanitization[ind] != 1:
                rnd = np.random.random()
                if rnd <= q2:
                    second_sanitization[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p2:
                    second_sanitization[ind] = 1

        return second_sanitization

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')


def L_SOUE_Aggregator_MI(reports, eps_perm, eps_1):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all L-SOUE sanitized UE-based vectors;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:

        # Number of reports
        n = len(reports)

        # SUE parameters for round 1
        p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        q1 = 1 - p1

        # OUE parameters for round 2
        p2 = 0.5
        q2 = (3.35410196624968 * (np.exp(eps_1) - 1) * (
                np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm)) * np.sqrt(
            0.0111111111111111 * np.exp(eps_1) + 0.00555555555555556 * np.exp(2 * eps_1) + 0.0444444444444444 * np.exp(
                0.5 * eps_perm) + 0.155555555555556 * np.exp(eps_perm) + 0.311111111111111 * np.exp(
                1.5 * eps_perm) + 0.388888888888889 * np.exp(2 * eps_perm) + 0.311111111111111 * np.exp(
                2.5 * eps_perm) + 0.155555555555556 * np.exp(3 * eps_perm) + 0.0444444444444444 * np.exp(
                3.5 * eps_perm) + 0.00555555555555556 * np.exp(4 * eps_perm) - 0.222222222222222 * np.exp(
                eps_1 + eps_perm) - 0.711111111111111 * np.exp(eps_1 + 1.5 * eps_perm) - np.exp(
                eps_1 + 2 * eps_perm) - 0.711111111111111 * np.exp(eps_1 + 2.5 * eps_perm) - 0.222222222222222 * np.exp(
                eps_1 + 3 * eps_perm) + 0.0111111111111111 * np.exp(eps_1 + 4 * eps_perm) + 0.0444444444444444 * np.exp(
                2 * eps_1 + 0.5 * eps_perm) + 0.155555555555556 * np.exp(2 * eps_1 + eps_perm) + 0.311111111111111 * np.exp(
                2 * eps_1 + 1.5 * eps_perm) + 0.388888888888889 * np.exp(
                2 * eps_1 + 2 * eps_perm) + 0.311111111111111 * np.exp(
                2 * eps_1 + 2.5 * eps_perm) + 0.155555555555556 * np.exp(
                2 * eps_1 + 3 * eps_perm) + 0.0444444444444444 * np.exp(
                2 * eps_1 + 3.5 * eps_perm) + 0.00555555555555556 * np.exp(
                2 * eps_1 + 4 * eps_perm) + 0.00555555555555556) - 0.25 * (
                      np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm) - np.exp(
                  eps_1 + 0.5 * eps_perm) - 2 * np.exp(eps_1 + eps_perm) - np.exp(eps_1 + 1.5 * eps_perm)) * (
                      np.exp(eps_1) + 4 * np.exp(0.5 * eps_perm) + 4 * np.exp(eps_perm) - np.exp(
                  2 * eps_perm) - 4 * np.exp(eps_1 + eps_perm) - 4 * np.exp(eps_1 + 1.5 * eps_perm) - np.exp(
                  eps_1 + 2 * eps_perm) + 1)) / (
                     (np.exp(eps_1) - 1) * (np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm)) * (
                     np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm) - np.exp(
                 eps_1 + 0.5 * eps_perm) - 2 * np.exp(eps_1 + eps_perm) - np.exp(eps_1 + 1.5 * eps_perm)))

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Count how many times each bit has been reported
        count_report = sum(reports)

        # Estimate with MI
        norm_est_freq = MI_long(count_report, n, p1, q1, p2, q2)

        return norm_est_freq

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')


def L_SOUE_Aggregator_IBU(reports, k, eps_perm, eps_1, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all L-SOUE UE-based vectors;
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

        # SUE parameters for round 1
        p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        q1 = 1 - p1

        # OUE parameters for round 2
        p2 = 0.5
        q2 = (3.35410196624968 * (np.exp(eps_1) - 1) * (
                np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm)) * np.sqrt(
            0.0111111111111111 * np.exp(eps_1) + 0.00555555555555556 * np.exp(2 * eps_1) + 0.0444444444444444 * np.exp(
                0.5 * eps_perm) + 0.155555555555556 * np.exp(eps_perm) + 0.311111111111111 * np.exp(
                1.5 * eps_perm) + 0.388888888888889 * np.exp(2 * eps_perm) + 0.311111111111111 * np.exp(
                2.5 * eps_perm) + 0.155555555555556 * np.exp(3 * eps_perm) + 0.0444444444444444 * np.exp(
                3.5 * eps_perm) + 0.00555555555555556 * np.exp(4 * eps_perm) - 0.222222222222222 * np.exp(
                eps_1 + eps_perm) - 0.711111111111111 * np.exp(eps_1 + 1.5 * eps_perm) - np.exp(
                eps_1 + 2 * eps_perm) - 0.711111111111111 * np.exp(eps_1 + 2.5 * eps_perm) - 0.222222222222222 * np.exp(
                eps_1 + 3 * eps_perm) + 0.0111111111111111 * np.exp(eps_1 + 4 * eps_perm) + 0.0444444444444444 * np.exp(
                2 * eps_1 + 0.5 * eps_perm) + 0.155555555555556 * np.exp(
                2 * eps_1 + eps_perm) + 0.311111111111111 * np.exp(
                2 * eps_1 + 1.5 * eps_perm) + 0.388888888888889 * np.exp(
                2 * eps_1 + 2 * eps_perm) + 0.311111111111111 * np.exp(
                2 * eps_1 + 2.5 * eps_perm) + 0.155555555555556 * np.exp(
                2 * eps_1 + 3 * eps_perm) + 0.0444444444444444 * np.exp(
                2 * eps_1 + 3.5 * eps_perm) + 0.00555555555555556 * np.exp(
                2 * eps_1 + 4 * eps_perm) + 0.00555555555555556) - 0.25 * (
                      np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm) - np.exp(
                  eps_1 + 0.5 * eps_perm) - 2 * np.exp(eps_1 + eps_perm) - np.exp(eps_1 + 1.5 * eps_perm)) * (
                      np.exp(eps_1) + 4 * np.exp(0.5 * eps_perm) + 4 * np.exp(eps_perm) - np.exp(
                  2 * eps_perm) - 4 * np.exp(eps_1 + eps_perm) - 4 * np.exp(eps_1 + 1.5 * eps_perm) - np.exp(
                  eps_1 + 2 * eps_perm) + 1)) / (
                     (np.exp(eps_1) - 1) * (np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm)) * (
                     np.exp(0.5 * eps_perm) + 2 * np.exp(eps_perm) + np.exp(1.5 * eps_perm) - np.exp(
                 eps_1 + 0.5 * eps_perm) - 2 * np.exp(eps_1 + eps_perm) - np.exp(eps_1 + 1.5 * eps_perm)))

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Count how many times each bit has been reported
        count_report = sum(reports)
        obs_freq = count_report / sum(count_report)

        # Estimate with IBU
        p = p1 * p2 + (1 - p1) * q2  # cf [1]
        q = (1 - q1) * q2 + q1 * p2  # cf [1]
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')

