import numpy as np
from pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from pure_frequency_oracles.UE import UE_Client, UE_Aggregator
from pure_frequency_oracles.Variance_PURE import VAR_Pure

def ADP_Client(input_data, k, epsilon, optimal=True):

    p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q_grr = (1 - p_grr) / (k - 1)
    var_grr = VAR_Pure(p_grr, q_grr)

    p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
    q_ue = 1 - p_ue

    if optimal:
        p_ue = 1 / 2
        q_ue = 1 / (np.exp(epsilon) + 1)

    var_ue = VAR_Pure(p_ue, q_ue)

    if var_grr <= var_ue:

        return GRR_Client(input_data, k, epsilon)
    else:

        return UE_Client(input_data, k, epsilon, optimal)


def ADP_Aggregator(reports, k, epsilon, optimal=True):

    p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q_grr = (1 - p_grr) / (k - 1)
    var_grr = VAR_Pure(p_grr, q_grr)

    p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
    q_ue = 1 - p_ue

    if optimal:
        p_ue = 1 / 2
        q_ue = 1 / (np.exp(epsilon) + 1)

    var_ue = VAR_Pure(p_ue, q_ue)

    if var_grr <= var_ue:

        return GRR_Aggregator(reports, k, epsilon)
    else:

        return UE_Aggregator(reports, epsilon, optimal)
