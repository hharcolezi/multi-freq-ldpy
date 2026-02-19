import numpy as np
from numba import jit

# [1] Agrawal et al (2001) "On the Design and Quantification of Privacy Preserving Data Mining Algorithms" (PODS).
# [2] Ehab et al (2020) "Generalized Iterative Bayesian Update and Applications to Mechanisms for Privacy Protection" (EuroS&P).
# [3] Pinzon et al (2026) "Estimating the True Distribution of Data Collected with Randomized Response" (AAAI).


def MI(count_report, n, p, q):
    """
    Matrix Inversion (MI).

    :param count_report : number of times that each value was reported;
    :param n : number of reports;
    :param p : probability of being honest;
    :param q : probability of lying;
    :return : normalized frequency (histogram) estimation.
    """

    # # Validations
    # if len(count_report) == 0:
    #     raise ValueError('List of count_report is empty.')
    # if not isinstance(n, int) or not isinstance(p, float) or not isinstance(q, float):
    #     raise ValueError('n (int), p (float), q (float) need numerical values.')

    # Ensure non-negativity of estimated frequency
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)

    # Re-normalized estimated frequency
    if sum(est_freq) > 0:
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    else:
        norm_est_freq = est_freq

    return norm_est_freq


def MI_long(count_report, n, p1, q1, p2, q2):
    """
    Matrix Inversion (MI).

    :param count_report : number of times that each value was reported;
    :param n : number of reports;
    :param p1 : probability of being honest in sanitization round 1;
    :param q1 : probability of lying in sanitization round 1;
    :param p2 : probability of being honest in sanitization round 2;
    :param q2 : probability of lying in sanitization round 2;
    :return : normalized frequency (histogram) estimation.
    """

    # # Validations
    # if len(count_report) == 0:
    #     raise ValueError('List of count_report is empty.')
    # if not isinstance(n, int) or not isinstance(p, float) or not isinstance(q, float):
    #     raise ValueError('n (int), p (float), q (float) need numerical values.')

    # Ensure non-negativity of estimated frequency
    est_freq = (
        (count_report - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))
    ).clip(0)

    # Re-normalized estimated frequency
    if sum(est_freq) > 0:
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

    else:
        norm_est_freq = est_freq

    return norm_est_freq


def MLE(count_report, n, p, q):
    """
    Maximum Likelihood Estimation (MLE). [3]

    :param count_report : number of times that each value was reported;
    :param n : number of reports;
    :param p : probability of being honest;
    :param q : probability of lying;
    :return : normalized frequency (histogram) estimation.
    """
    assert p >= q >= 0, ("Invalid noise parameters", p, q)
    assert np.isclose(np.sum(count_report), n), \
        ("Invalid count_report", count_report, n)
    phi = np.asarray(count_report) / n
    n_cats = len(phi)
    sig = np.argsort(phi)
    k = 0
    s = 1  # invariant: s = sum(phi[sig[k:]])
    while k < n_cats and q * s > (1 - k * q) * phi[sig[k]]:
        s -= phi[sig[k]]
        k += 1
    theta = np.zeros(n_cats)
    theta[sig[k:]] = ((1 - k * q) * phi[sig[k:]] - s * q) / (s * (p - q))

    # Renormalize just for numerical stability (in theory, the sum should be 1)
    theta = np.nan_to_num(theta / np.sum(theta))
    return theta


@jit(nopython=True)
def IBU(k, A, obs_freq, nb_iter, tol, err_func):
    """
    Iterative Bayesian Update (IBU)[1,2].

    :param k : attribute's domain size;
    :param A : probability matrix;
    :param obs_freq : observed frequency;
    :param nb_iter : number of iterations;
    :param tol : tolerance;
    :param err_func: early stopping function;
    :return : frequency estimation.
    """

    # # Validations
    # if len(obs_freq) == 0:
    #     raise ValueError('List of obs_freq is empty.')
    # if not all(isinstance(l, np.ndarray) for l in A):
    #     raise ValueError('A must be a np.ndarray.')
    # if not isinstance(k, int) or not isinstance(nb_iter, int):
    #     raise ValueError('k and nb_iter need integer values.')
    # if nb_iter<=0 or tol<=0:
    #     raise ValueError('nb_iter (int) and tol (float) need values greater than 0')

    # Step 1 - Expectation: initialization of probabilities as a uniform distribution
    est_freq = np.ones(k) / k
    est_freq_t = None

    # Step 2 - Maximization: calculating estimated frequencies
    if err_func == "max_abs":
        for _ in range(nb_iter):
            est_freq_t = est_freq * np.dot(A, obs_freq / np.dot(A, est_freq))
            if np.abs(est_freq - est_freq_t).max() < tol:
                return est_freq_t
            else:
                est_freq = est_freq_t

    elif err_func == "mse":
        for _ in range(nb_iter):
            est_freq_t = est_freq * np.dot(A, obs_freq / np.dot(A, est_freq))
            if np.square(np.subtract(est_freq_t, est_freq)).mean() < tol:
                return est_freq_t
            else:
                est_freq = est_freq_t

    elif err_func == "mae":
        for _ in range(nb_iter):
            est_freq_t = est_freq * np.dot(A, obs_freq / np.dot(A, est_freq))
            if np.abs(est_freq - est_freq_t).mean() < tol:
                return est_freq_t
            else:
                est_freq = est_freq_t

    elif err_func == "max_squared":
        for _ in range(nb_iter):
            est_freq_t = est_freq * np.dot(A, obs_freq / np.dot(A, est_freq))
            if np.square(np.subtract(est_freq_t, est_freq)).max() < tol:
                return est_freq_t
            else:
                est_freq = est_freq_t
    else:
        raise ValueError(
            "Error function unknown. Options are: max_abs, mse, mae, max_squared."
        )

    return est_freq_t
