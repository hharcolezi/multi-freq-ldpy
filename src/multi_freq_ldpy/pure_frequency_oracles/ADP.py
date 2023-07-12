import numpy as np
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator_MI
from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client, UE_Aggregator_MI
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator_IBU
from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client, UE_Aggregator_IBU
from multi_freq_ldpy.pure_frequency_oracles.Variance_PURE import VAR_Pure

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).

def ADP_Client(input_data, k, epsilon, optimal=True):

    """
    Adaptive (ADP) protocol that minimizes variance value (i.e., either GRR or OUE from [1]).

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: sanitized value or UE vector.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # GRR parameters
        p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # Symmetric parameters (p+q = 1)
        p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q_ue = 1 - p_ue

        # Optimized parameters
        if optimal:
            p_ue = 1 / 2
            q_ue = 1 / (np.exp(epsilon) + 1)

        # Variance values
        var_grr = VAR_Pure(p_grr, q_grr)
        var_ue = VAR_Pure(p_ue, q_ue)

        # Adaptive protocol
        if var_grr <= var_ue:

            return GRR_Client(input_data, k, epsilon)
        else:

            return UE_Client(input_data, k, epsilon, optimal)

    else:
        raise ValueError('k (int) and epsilon (float) need a numerical value.')


def ADP_Aggregator_MI(reports, k, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all GRR sanitized values or UE-based vectors;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(k, int) or k < 2:
        raise ValueError('k needs an integer value >=2.')
    if epsilon > 0:

        # GRR parameters
        p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # Symmetric parameters (p+q = 1)
        p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q_ue = 1 - p_ue

        # Optimized parameters
        if optimal:
            p_ue = 1 / 2
            q_ue = 1 / (np.exp(epsilon) + 1)

        # Variance values
        var_grr = VAR_Pure(p_grr, q_grr)
        var_ue = VAR_Pure(p_ue, q_ue)

        # Adaptive estimator
        if var_grr <= var_ue:

            return GRR_Aggregator_MI(reports, k, epsilon)
        else:

            return UE_Aggregator_MI(reports, epsilon, optimal)

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')


def ADP_Aggregator_IBU(reports, k, epsilon, optimal=True, nb_iter=10000, tol=1e-12, err_func="max_abs"):

    """
    Estimator based on Iterative Bayesian Update.

    :param reports: list of all GRR sanitized values or UE-based vectors;
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

        # GRR parameters
        p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # Symmetric parameters (p+q = 1)
        p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q_ue = 1 - p_ue

        # Optimized parameters
        if optimal:
            p_ue = 1 / 2
            q_ue = 1 / (np.exp(epsilon) + 1)

        # Variance values
        var_grr = VAR_Pure(p_grr, q_grr)
        var_ue = VAR_Pure(p_ue, q_ue)

        # Adaptive estimator
        if var_grr <= var_ue:

            return GRR_Aggregator_IBU(reports, k, epsilon, nb_iter, tol, err_func)
        else:

            return UE_Aggregator_IBU(reports, k, epsilon, optimal, nb_iter, tol, err_func)

    else:
        raise ValueError('epsilon needs a numerical value greater than 0.')
