import numpy as np
from numba import jit
from multi_freq_ldpy.estimators.Histogram_estimator import MI, IBU

# [1] Ye and Barg (2018) "Optimal schemes for discrete distribution estimation under locally differential privacy" (IEEE Transactions on Information Theory).
# [2] Wang et al (2016) "Mutual information optimally local private discrete distribution estimation" (arXiv:1607.08025).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

@jit(nopython=True)
def SS_Client(input_data, k, epsilon):
    """
    Subset Selection (SS) protocol [1,2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: set of sub_k sanitized values.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # Mapping domain size k to the range [0, ..., k-1]
        domain = np.arange(k)

        # SS parameters
        sub_k = int(max(1, np.rint(k / (np.exp(epsilon) + 1))))
        p_v = sub_k * np.exp(epsilon) / (sub_k * np.exp(epsilon) + k - sub_k)

        # SS perturbation function
        rnd = np.random.random()
        sub_set = np.zeros(sub_k, dtype='int64')
        if rnd <= p_v:
            sub_set[0] = int(input_data)
            sub_set[1:] = np.random.choice(domain[domain != input_data], size=sub_k-1, replace=False)
            return sub_set

        else:
            return np.random.choice(domain[domain != input_data], size=sub_k, replace=False)

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')

def SS_Aggregator_MI(reports, k, epsilon):
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

        # SS parameters
        sub_k = int(max(1, np.rint(k / (np.exp(epsilon) + 1))))
        p = sub_k * np.exp(epsilon) / (sub_k * np.exp(epsilon) + k - sub_k)
        q = ((sub_k - 1) * (sub_k * np.exp(epsilon)) + (k - sub_k) * sub_k) / ((k - 1) * (sub_k * np.exp(epsilon) + k - sub_k))

        # Count how many times each value has been reported
        count_report = np.zeros(k)
        for rep in reports:
            for i in range(sub_k):
                count_report[rep[i]] += 1

        # Estimate with MI
        norm_est_freq = MI(count_report, n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')


def SS_Aggregator_IBU(reports, k, epsilon, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all SS-based sanitized values;
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

        # Number of reports
        n = len(reports)

        # SS parameters
        sub_k = int(max(1, np.rint(k / (np.exp(epsilon) + 1))))
        p = sub_k * np.exp(epsilon) / (sub_k * np.exp(epsilon) + k - sub_k)
        q = ((sub_k - 1) * (sub_k * np.exp(epsilon)) + (k - sub_k) * sub_k) / (
                    (k - 1) * (sub_k * np.exp(epsilon) + k - sub_k))
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q

        # Count how many times each value has been reported
        obs_count = np.zeros(k)
        for rep in reports:
            for i in range(sub_k):
                obs_count[rep[i]] += 1
        obs_freq = obs_count / sum(obs_count)

        # Estimate with IBU
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('epsilon (float) need a numerical value greater than 0.')