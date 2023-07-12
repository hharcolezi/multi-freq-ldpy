import numpy as np
from numba import jit

# [1] Ding, Kulkarni, and Yekhanin (2017) "Collecting telemetry data privately" (NeurIPS).

@jit(nopython=True)
def dBitFlipPM_Client(input_data, k, b, d_bits, eps_perm):
    
    """
    dBitFlipPM [1] protocol that applies a single round of sanitization (permanent memoization).

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param b: new domain size to map k to b buckets;
    :param d_bits: number of bits to report;
    :param eps_perm: upper bound of privacy guarantee;
    :return: sanitized UE vector.
    """

    # Validations
    if input_data < 0 or input_data >= k:
        raise ValueError('input_data (integer) should be in the range [0, k-1].')
    if not isinstance(k, int) or not isinstance(b, int) or not isinstance(d_bits, int):
        raise ValueError('k, b, and d_bits need integer values.')
    if b > k or d_bits > b:
        raise ValueError('Please ensure k, b, and d_bits follow d_bits <= b <= k.')
    if eps_perm > 0 and b > 0 and d_bits > 0:

        # SUE parameters
        p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        q1 = 1 - p1

        # calculate bucket size
        bucket_size = k / b

        # select d_bits random bits j_1, j_2, ..., j_d (without replacement)
        j = np.random.choice(np.arange(b), size=d_bits, replace=False)

        # calculate the buck number of user's value
        bucketized_data = int(input_data / bucket_size)

        # Unary encoding
        permanent_sanitization = np.ones(b) * - 1 # set to -1 non-sampled bits

        # Permanent Memoization
        idx_j = 0
        for i in range(b):
            if i in j: # only the sampled bits
                rand = np.random.random()
                if bucketized_data == j[idx_j]:
                    permanent_sanitization[j[idx_j]] = int(rand <= p1)
                else:
                    permanent_sanitization[j[idx_j]] = int(rand <= q1)

                idx_j+=1

        return permanent_sanitization

    else:
        raise ValueError('Please ensure eps_perm, b, and d_bits have numerical values greater than 0.')

def dBitFlipPM_Aggregator_MI(reports, b, d_bits, eps_perm):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all dBitFlipPM sanitized UE vectors;
    :param b: new domain size to map k to b buckets;
    :param d_bits: number of bits to report;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :return: normalized frequency (histogram) estimation.
    """

    # Validations
    if len(reports) == 0:
        raise ValueError('List of reports is empty.')
    if not isinstance(b, int) or not isinstance(d_bits, int):
        raise ValueError('k and d_bits need integer values.')
    if eps_perm > 0:
    
        # Estimated frequency of each bucket
        est_freq = []
        for v in range(b):
            h = 0
            for bi in reports:
                if bi[v] >= 0: # only the sampled bits
                    h += (bi[v] * (np.exp(eps_perm / 2) + 1) - 1) / (np.exp(eps_perm / 2) - 1)
            est_freq.append(h * b / (len(reports) * d_bits ))

        # Ensure non-negativity of estimated frequency
        est_freq = np.array(est_freq).clip(0)

        # Re-normalized estimated frequency
        if sum(est_freq) > 0:
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

        else:
            norm_est_freq = est_freq

        return norm_est_freq

    else:
        raise ValueError('eps_perm needs a numerical value greater than 0.')
