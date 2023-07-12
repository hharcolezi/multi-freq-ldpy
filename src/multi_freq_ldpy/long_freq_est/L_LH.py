import numpy as np
import xxhash
from sys import maxsize
from multi_freq_ldpy.long_freq_est.L_GRR import L_GRR_Client
from multi_freq_ldpy.estimators.Histogram_estimator import MI_long, IBU

# [1] Arcolezi et al (2023) "Frequency Estimation of Evolving Data Under Local Differential Privacy" (EDBT 2023).
# [2] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).

def L_LH_Client(input_data, k, eps_perm, eps_1, optimal=True):
    
    """
    Longitudinal Local Hashing (L-LH) [1] protocol.

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :param optimal: if True, it uses the Optimized Longitudinal LH (L-OLH) protocol from [1];
    :return: tuple of sanitized value and random seed.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:
    
        # L-BLH parameter
        g = 2

        # L-OLH parameter
        if optimal:
            alpha = eps_1/eps_perm
            g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))

        # Generate random "hash function" (i.e., seed) and hash the user's value
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

        # L-LH perturbation function (i.e., L-GRR-based)
        sanitized_value = L_GRR_Client(hashed_input_data, g, eps_perm, eps_1)

        return (sanitized_value, rnd_seed)

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')

def L_LH_Aggregator_MI(reports, k, eps_perm, eps_1, optimal=True):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all LH-based sanitized values;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :param optimal: if True, it uses the Optimized Longitudinal LH (L-OLH) protocol from [1];
    :return: frequency estimation.
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
        
        # L-BLH parameter
        g = 2
        
        # L-OLH parameter
        if optimal:
            alpha = eps_1/eps_perm
            g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))

        # GRR parameters for round 1
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
        q1 = (1 - p1) / (g-1)

        # GRR parameters for round 2
        p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + g*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(g-1)+q1)
        q2 = (1 - p2) / (g-1)
        
        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')
        
        # Count how many times each value has been reported
        obs_count = np.zeros(k)
        for tuple_val_seed in reports:
            for v in range(k):
                if tuple_val_seed[0] == (xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % g):
                    obs_count[v] += 1
        
        # Estimate with MI
        q1 = 1 / g  # updating q1 in the server
        norm_est_freq = MI_long(obs_count, n, p1, q1, p2, q2)

        return norm_est_freq
        
    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')
  

def L_LH_Aggregator_IBU(reports, k, eps_perm, eps_1, optimal=True, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all LH-based sanitized values;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :param optimal: if True, it uses the Optimized Longitudinal LH (L-OLH) protocol from [1];
    :param nb_iter: number of iterations;
    :param tol: tolerance;
    :param err_func: early stopping function;
    :return: frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if eps_perm < 0 or eps_1 < 0:
        raise ValueError('Please ensure eps_perm and eps_1 have numerical values greater than 0.')
    if eps_1 < eps_perm:

        # L-BLH parameter
        g = 2

        # L-OLH parameter
        if optimal:
            alpha = eps_1 / eps_perm
            g = int(max(np.rint((np.sqrt(np.exp(4 * eps_perm) - 14 * np.exp(2 * eps_perm) - 12 * np.exp(
                2 * eps_perm * (alpha + 1)) + 12 * np.exp(eps_perm * (alpha + 1)) + 12 * np.exp(
                eps_perm * (alpha + 3)) + 1) - np.exp(2 * eps_perm) + 6 * np.exp(eps_perm) - 6 * np.exp(
                eps_perm * alpha) + 1) / (6 * (np.exp(eps_perm) - np.exp(eps_perm * alpha)))), 2))

        # GRR parameters for round 1
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
        q1 = (1 - p1) / (g - 1)

        # GRR parameters for round 2
        p2 = (q1 - np.exp(eps_1) * p1) / (
                    (-p1 * np.exp(eps_1)) + g * q1 * np.exp(eps_1) - q1 * np.exp(eps_1) - p1 * (g - 1) + q1)
        q2 = (1 - p2) / (g - 1)

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Count how many times each value has been reported
        obs_count = np.zeros(k)
        for tuple_val_seed in reports:
            for v in range(k):
                if tuple_val_seed[0] == (xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % g):
                    obs_count[v] += 1
        obs_freq = obs_count / sum(obs_count)

        # Estimate with IBU
        q1 = 1 / g  # updating q1 in the server
        p = p1 * p2 + (1 - p1) * q2  # cf [2]
        q = p1 * q2 + q1 * p2  # cf [2]
        A = np.eye(k)
        A[A == 1] = p
        A[A == 0] = q
        est_freq = IBU(k, A, obs_freq, nb_iter, tol, err_func)

        return est_freq

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')
