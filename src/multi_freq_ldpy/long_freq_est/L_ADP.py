import numpy as np
from multi_freq_ldpy.long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator_MI
from multi_freq_ldpy.long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator_MI
from multi_freq_ldpy.long_freq_est.Variance_LONG_PURE import VAR_Long_Pure

# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).

# The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from: https://github.com/hharcolezi/ldp-protocols-mobility-cdrs/blob/main/papers/%5B4%5D/1_ALLOMFREE_Analysis.ipynb

def L_ADP_Client(input_data, k, eps_perm, eps_1):

    """
    Longitudinal Adaptive (L-ADP) protocol that minimizes variance value (i.e., either L-GRR or L-OSUE from [1], a.k.a. ALLOMFREE).

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: sanitized value or UE vector.
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
        p1_grr = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1_grr = (1 - p1_grr) / (k - 1)

        #  GRR parameters for round 2
        p2_grr = (q1_grr - np.exp(eps_1) * p1_grr) / (
                    (-p1_grr * np.exp(eps_1)) + k * q1_grr * np.exp(eps_1) - q1_grr * np.exp(eps_1) - p1_grr * (k - 1) + q1_grr)
        q2_grr = (1 - p2_grr) / (k - 1)

        if (np.array([p1_grr, q1_grr, p2_grr, q2_grr]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # OUE parameters for round 1
        p1_ue = 1 / 2
        q1_ue = 1 / (np.exp(eps_perm) + 1)

        # SUE parameters for round 2
        p2_ue = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
        q2_ue = 1 - p2_ue

        if (np.array([p1_ue, q1_ue, p2_ue, q2_ue]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Variance values of L-GRR and L-OSUE
        var_l_grr = VAR_Long_Pure(p1_grr, q1_grr, p2_grr, q2_grr)
        var_l_osue = VAR_Long_Pure(p1_ue, q1_ue, p2_ue, q2_ue)

        # Adaptive longitudinal protocol (a.k.a. ALLOMFREE in [1])
        if var_l_grr <= var_l_osue:

            return L_GRR_Client(input_data, k, eps_perm, eps_1)
        else:

            return L_OSUE_Client(input_data, k, eps_perm, eps_1)

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')


def L_ADP_Aggregator_MI(reports, k, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all sanitized L-GRR values or L-OSUE UE-based vectors;
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

        # GRR parameters for round 1
        p1_grr = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1_grr = (1 - p1_grr) / (k - 1)

        #  GRR parameters for round 2
        p2_grr = (q1_grr - np.exp(eps_1) * p1_grr) / (
                (-p1_grr * np.exp(eps_1)) + k * q1_grr * np.exp(eps_1) - q1_grr * np.exp(eps_1) - p1_grr * (k - 1) + q1_grr)
        q2_grr = (1 - p2_grr) / (k - 1)

        if (np.array([p1_grr, q1_grr, p2_grr, q2_grr]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # OUE parameters for round 1
        p1_ue = 1 / 2
        q1_ue = 1 / (np.exp(eps_perm) + 1)

        # SUE parameters for round 2
        p2_ue = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
        q2_ue = 1 - p2_ue

        if (np.array([p1_ue, q1_ue, p2_ue, q2_ue]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Variance values of L-GRR and L-OSUE
        var_l_grr = VAR_Long_Pure(p1_grr, q1_grr, p2_grr, q2_grr)
        var_l_osue = VAR_Long_Pure(p1_ue, q1_ue, p2_ue, q2_ue)

        # Adaptive estimator
        if var_l_grr <= var_l_osue:

            return L_GRR_Aggregator_MI(reports, k, eps_perm, eps_1)

        else:
            return L_OSUE_Aggregator_MI(reports, eps_perm, eps_1)

    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')

def L_ADP_Aggregator_IBU(reports, k, eps_perm, eps_1, nb_iter=10000, tol=1e-12, err_func="max_abs"):
    """
    Estimator based on Iterative Bayesian Update[3,4].

    :param reports: list of all L-GRR sanitized values or UE-based vectors;
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
        p1_grr = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1_grr = (1 - p1_grr) / (k - 1)

        #  GRR parameters for round 2
        p2_grr = (q1_grr - np.exp(eps_1) * p1_grr) / (
                (-p1_grr * np.exp(eps_1)) + k * q1_grr * np.exp(eps_1) - q1_grr * np.exp(eps_1) - p1_grr * (
                    k - 1) + q1_grr)
        q2_grr = (1 - p2_grr) / (k - 1)

        if (np.array([p1_grr, q1_grr, p2_grr, q2_grr]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # OUE parameters for round 1
        p1_ue = 1 / 2
        q1_ue = 1 / (np.exp(eps_perm) + 1)

        # SUE parameters for round 2
        p2_ue = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
        q2_ue = 1 - p2_ue

        if (np.array([p1_ue, q1_ue, p2_ue, q2_ue]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative.')

        # Variance values of L-GRR and L-OSUE
        var_l_grr = VAR_Long_Pure(p1_grr, q1_grr, p2_grr, q2_grr)
        var_l_osue = VAR_Long_Pure(p1_ue, q1_ue, p2_ue, q2_ue)

        # Adaptive estimator
        if var_l_grr <= var_l_osue:

            return L_GRR_Aggregator_IBU(reports, k, eps_perm, eps_1, nb_iter, tol, err_func)

        else:
            return L_OSUE_Aggregator_IBU(reports, eps_perm, eps_1, nb_iter, tol, err_func)
    else:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')
