# Multi-Freq-LDPy: Multiple Frequency Estimation Under Local Differential Privacy in Python

Multi-Freq-LDPy is a Python library for performing multiple frequency estimation tasks (multidimensional, longitudinal, and both) under local differential privacy (LDP) guarantees. The main goal is to provide an easy-to-use and fast execution toolkit to benchmark and experiment with state-of-the-art solutions and LDP protocols.

Here's an introductory [Video_Presentation](https://screencast-o-matic.com/watch/c3hhQYVYNDi) and [Slide_Presentation](http://hharcolezi.github.io/files/2022_Multi_Freq_LDPy_Presentation.pdf) of our package.

## Installation

Please use the package manager [pip](https://pip.pypa.io/en/stable/) to install multi-freq-ldpy.

```bash
pip install multi-freq-ldpy
```

To ensure you use the [latest version](https://pypi.org/project/multi-freq-ldpy/).

```bash
pip install multi-freq-ldpy --upgrade
```

Multi-Freq-LDPy requires the following Python packages.

```bash
numpy
numba
xxhash
```

## Content
Multi-Freq-LDPy covers the following tasks: 

1. **Single Frequency Estimation** -- The best-performing frequency oracles from [Locally Differentially Private Protocols for Frequency Estimation](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao), namely:
   * Generalized Randomized Response (GRR): ```multi_freq_ldpy.pure_frequency_oracles.GRR```
   * Symmetric/Optimized Unary Encoding (UE): ```multi_freq_ldpy.pure_frequency_oracles.UE```
   * Binary/Optimized Local Hashing (LH): ```multi_freq_ldpy.pure_frequency_oracles.LH```
   * Adaptive (ADP) protocol, i.e., GRR or Optimized UE: ```multi_freq_ldpy.pure_frequency_oracles.ADP```

2. **Multidimensional Frequency Estimation** -- Three solutions for frequency estimation of multiple attributes from [Random Sampling Plus Fake Data: Multidimensional Frequency Estimates With Local Differential Privacy](https://arxiv.org/abs/2109.07269) with their respective frequency oracles (GRR, UE-based, and ADP), namely:
   * Splitting (SPL) the privacy budget: ```multi_freq_ldpy.mdim_freq_est.SPL_solution```
   * Random Sampling (SMP) a single attribute: ```multi_freq_ldpy.mdim_freq_est.SMP_solution```
   * Random Sampling + Fake Data (RS+FD) that samples a single attribute but also generates fake data for each non-sampled attribute: ```multi_freq_ldpy.mdim_freq_est.RSpFD_solution```

3. **Longitudinal Single Frequency Estimation** -- All longitudinal LDP protocols from [Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates](https://arxiv.org/abs/2111.04636) following the memoization-based framework from [RAPPOR](https://dl.acm.org/doi/10.1145/2660267.2660348), namely:
   * Longitudinal GRR (L-GRR): ```multi_freq_ldpy.long_freq_est.L_GRR```
   * Longitudinal OUE (L-OUE): ```multi_freq_ldpy.long_freq_est.L_OUE```
   * Longitudinal OUE-SUE (L-OSUE): ```multi_freq_ldpy.long_freq_est.L_OSUE```
   * Longitudinal SUE (L-SUE): ```multi_freq_ldpy.long_freq_est.L_SUE```
   * Longitudinal SUE-OUE (L-SOUE): ```multi_freq_ldpy.long_freq_est.L_SOUE```
   * Longitudinal ADP (L-ADP), i.e., L-GRR or L-OSUE: ```multi_freq_ldpy.long_freq_est.L_ADP```

4. **Longitudinal Multidimensional Frequency Estimation** -- Both SPL and SMP solutions with all longitudinal protocols from previous point 3, namely:
   * Longitudinal SPL (L_SPL): ```multi_freq_ldpy.long_mdim_freq_est.L_SPL```
   * Longitudinal SMP (L_SMP): ```multi_freq_ldpy.long_mdim_freq_est.L_SMP```

## Usage
This is a function-based package that simulates the LDP data collection pipeline of users and the server. For each functionality, there is always a ```Client``` and an ```Aggregator``` function. For more details, please refer to the [tutorials](https://github.com/hharcolezi/multi-freq-ldpy/tree/main/tutorials) folder, which covers all 1--4 tasks with real-world open datasets ([Adult](https://archive.ics.uci.edu/ml/datasets/adult), [Nursery](https://archive.ics.uci.edu/ml/datasets/nursery), [MS-FIMU](https://github.com/hharcolezi/OpenMSFIMU)).

```python
# Common libraries
import numpy as np
import matplotlib.pyplot as plt

# Multi-Freq-LDPy functions for L-SUE protocol (a.k.a. Basic RAPPOR)
from multi_freq_ldpy.long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator

# Parameters for simulation
epsilon_perm = 2 # longitudinal privacy guarantee, i.e., upper bound (infinity reports)
epsilon_1 = 0.5 * epsilon_perm # single report privacy guarantee, i.e., lower bound
n = int(1e6) # number of users
k = 5 # attribute's domain size

# Simulation dataset where every user has a number between [0-k) with n users
data = np.random.randint(k, size=n)

# Simulation of client-side
l_sue_reports = [L_SUE_Client(input_data, k, epsilon_perm, epsilon_1) for input_data in data]

# Simulation of server-side aggregation
l_sue_est_freq = L_SUE_Aggregator(l_sue_reports, epsilon_perm, epsilon_1)

# Real frequency 
real_freq = np.unique(data, return_counts=True)[-1] / n

# Visualizing results
barwidth = 0.45
x_axis = np.arange(k)

plt.bar(x_axis - barwidth/2, real_freq, label='Real Freq', width=barwidth)
plt.bar(x_axis + barwidth/2 , l_sue_est_freq, label='Est Freq: L-SUE', width=barwidth)
plt.ylabel('Normalized Frequency')
plt.xlabel('Domain values')
plt.legend(loc='upper right', bbox_to_anchor=(1.015, 1.15))
plt.show();
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact
For any question, please contact [Heber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr

## Acknowledgments
   * The Local Hashing (LH) functions were adapted from the [pure-LDP](https://github.com/Samuel-Maddock/pure-LDP) package, which covers a wider range of frequency oracles for single-frequency estimation.
   * Some codes were adapted from our [ldp-protocols-mobility-cdrs](https://github.com/hharcolezi/ldp-protocols-mobility-cdrs) repository. 

## License
[MIT](https://github.com/hharcolezi/multi-freq-ldpy/blob/main/LICENSE)