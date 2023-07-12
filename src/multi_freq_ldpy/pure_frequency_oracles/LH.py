import numpy as np
from sys import maxsize
import xxhash
from multi_freq_ldpy.estimators.Histogram_estimator import MI, IBU
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Bassily and Smith "Local, private, efficient protocols for succinct histograms" (STOC).
# [3] Agrawal and Aggarwal (2001,) "On the design and quantification of privacy preserving data mining algorithms" (PODS).
# [4] ElSalamouny and Palamidessi (2020) "Generalized iterative bayesian update and applications to mechanisms for privacy protection" (EuroS&P).

# Code adapted from pure-ldp library (https://github.com/Samuel-Maddock/pure-LDP) developed by Samuel Maddock

def LH_Client(input_data, k, epsilon, optimal=True):
    
    """
    Local Hashing (LH) protocol[1], which is logically equivalent to the random matrix projection technique in [2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [1];
    :return: tuple of sanitized value and random seed.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # Binary LH (BLH) parameter
        g = 2

        # Optimal LH (OLH) parameter
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # Generate random seed and hash the user's value
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

        # LH perturbation function (i.e., GRR-based)
        sanitized_value = GRR_Client(hashed_input_data, g, epsilon)

        return (sanitized_value, rnd_seed)

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')

def LH_Aggregator_MI(reports, k, epsilon, optimal=True):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all LH-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [1];
    :return: frequency estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:
            
        # Number of reports
        n = len(reports)

        # Binary LH (BLH) parameter
        g = 2

        # Optimal LH (OLH) parameter
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # Count how many times each value has been reported
        count_report = np.zeros(k)
        for tuple_val_seed in reports:
            for v in range(k):
                if tuple_val_seed[0] == (xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % g):
                    count_report[v] += 1

        # Estimate with MI
        p = np.exp(epsilon) / (np.exp(epsilon) + g - 1) # GRR parameters with reduced domain size g
        q = 1 / g # g value differs at the server side [1]
        norm_est_freq = MI(count_report, n, p, q)

        return norm_est_freq

    else:
        raise ValueError('epsilon (float) needs a numerical value greater than 0.')


def LH_Aggregator_IBU(reports, k, epsilon, optimal=True, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all LH-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [1];
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

        # Binary LH (BLH) parameter
        g = 2

        # Optimal LH (OLH) parameter
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # Count how many times each value has been reported
        obs_count = np.zeros(k)
        for tuple_val_seed in reports:
            for v in range(k):
                if tuple_val_seed[0] == (xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % g):
                    obs_count[v] += 1
        obs_freq = obs_count / sum(obs_count)

        # Estimate with IBU
        p = np.exp(epsilon) / (np.exp(epsilon) + g - 1) # GRR parameters with reduced domain size g
        q = 1 / g  # g value differs at the server side [1]
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('epsilon (float) need a numerical value greater than 0.')