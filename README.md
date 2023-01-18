# Multi-Freq-LDPy: Multiple Frequency Estimation Under Local Differential Privacy in Python

Multi-Freq-LDPy is a Python library for performing multiple frequency estimation tasks (multidimensional, longitudinal, and both) under local differential privacy (LDP) guarantees. The main goal is to provide an easy-to-use and fast execution toolkit to benchmark and experiment with state-of-the-art solutions and LDP protocols.

Here's an introductory [Video_Presentation](https://screencast-o-matic.com/watch/c3hhQYVYNDi), [Slide_Presentation](http://hharcolezi.github.io/files/2022_Multi_Freq_LDPy_Presentation.pdf), and [arXived Demonstration Paper](https://arxiv.org/abs/2205.02648) of our package.

If our codes and work are useful to you, we would appreciate a reference to:

```
@inproceedings{Arcolezi2022,
  author="Arcolezi, H{\'e}ber H.
  and Couchot, Jean-Fran{\c{c}}ois
  and Gambs, S{\'e}bastien
  and Palamidessi, Catuscia
  and Zolfaghari, Majid",
  editor="Atluri, Vijayalakshmi
  and Di Pietro, Roberto
  and Jensen, Christian D.
  and Meng, Weizhi",
  title="Multi-Freq-LDPy: Multiple Frequency Estimation Under Local Differential Privacy in Python",
  booktitle="Computer Security -- ESORICS 2022",
  year="2022",
  doi="10.1007/978-3-031-17143-7_40",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="770--775",
}
```

## Installation

Please use the package manager [pip](https://pypi.org/project/multi-freq-ldpy/) to install multi-freq-ldpy.

```bash
pip install multi-freq-ldpy
```

To ensure you use the latest version.

```
pip install multi-freq-ldpy --upgrade
```

Multi-Freq-LDPy requires the following Python packages.

```
numpy
numba
xxhash
```

## Content
Multi-Freq-LDPy and its modules are structured as follows. 

```
multi-freq-ldpy package
|
|- pure_frequency_oracles (Single Frequency Estimation)
|  |- GRR (Generalized Randomized Response[1,2] a.k.a. k-RR or Direct Encoding)
|  |- UE (Unary Encoding)
|  |  |- SUE (Symmetric UE[3] a.k.a. Basic One-Time RAPPOR[11])
|  |  |- OUE (Optimized UE[3])
|  |- LH (Local Hashing)
|  |  |- BLH (Binary LH[3,4])
|  |  |- OLH (Optimized LH[3])
|  |- SS (Subset Selection[5,6])
|  |- ADP (Adaptive, i.e., GRR or OUE)
|
|- mdim_freq_est (Multidimensional Frequency Estimation)
|  |- SPL_solution (Splitting solution[7,8]): Splits the privacy budget and sanitizes using pure_frequency_oracles LDP protocols
|  |  |- SPL_GRR, SPL_SUE, SPL_OUE, SPL_BLH, SPL_OLH, SPL_SS, SPL_ADP
|  |- SMP_solution (Random Sampling solution[7,8]): Samples a single attribute and sanitizes using pure_frequency_oracles LDP protocols
|  |  |- SMP_GRR, SMP_SUE, SMP_OUE, SMP_BLH, SMP_OLH, SMP_SS, SMP_ADP
|  |- RSpFD_solution (Random Sampling + Fake Data solution[9]): Samples a single attribute to sanitize but also generates fake data for each non-sampled attribute
|  |  |- RSpFD_GRR (fake data generated following domain size)
|  |  |- RSpFD_SUE_zero (fake data generated with SUE applied to a zero-vector)
|  |  |- RSpFD_SUE_rnd (fake data generated with SUE applied to a random bit-vector)
|  |  |- RSpFD_OUE_zero (fake data generated with OUE applied to a zero-vector)
|  |  |- RSpFD_OUE_rnd (fake data generated with OUE applied to a random bit-vector)
|  |  |- RSpFD_ADP (RSpFD_GRR or RSpFD_OUE_z)
|
|- long_freq_est (Longitudinal Single Frequency Estimation)
|  |- L_GRR (Longitudinal GRR[10])
|  |- L_OUE (Longitudinal OUE[10])
|  |- L_OSUE (Longitudinal OUE-SUE[10])
|  |- L_SUE (Longitudinal SUE[10], a.k.a. Basic RAPPOR[11])
|  |- L_SOUE (Longitudinal SUE-OUE[10])
|  |- L_ADP (Longitudinal ADP[10], i.e., L-GRR or L-OSUE)
|  |- dBitFlipPM[12]
|
|- long_mdim_freq_est (Longitudinal Multidimensional Frequency Estimation)
|  |- Longitudinal SPL (L_SPL_Solution[10]): Splits the privacy budget and sanitizes using long_freq_est LDP protocols
|  |  |- SPL_L_GRR, SPL_L_OUE, SPL_L_OSUE, SPL_L_SUE, SPL_L_SOUE, SPL_L_ADP, SPL_dBitFlipPM
|  |- Longitudinal SMP (L_SMP_Solution[10]): Samples a single attribute and sanitizes using long_freq_est LDP protocols
|  |  |- SMP_L_GRR, SMP_L_OUE, SMP_L_OSUE, SMP_L_SUE, SMP_L_SOUE, SMP_L_ADP, SMP_dBitFlipPM
```

## Usage
This is a function-based package that simulates the LDP data collection pipeline of users and the server. For each functionality, there is always a ```Client``` and an ```Aggregator``` function. For more details, please refer to the [tutorials](https://github.com/hharcolezi/multi-freq-ldpy/tree/main/tutorials) folder, which covers all 1--4 tasks with real-world open datasets ([Adult](https://archive.ics.uci.edu/ml/datasets/adult), [Nursery](https://archive.ics.uci.edu/ml/datasets/nursery), [MS-FIMU](https://github.com/hharcolezi/OpenMSFIMU)).

```python
# Common libraries
import numpy as np
import matplotlib.pyplot as plt

# Multi-Freq-LDPy functions for L-SUE protocol (a.k.a. Basic RAPPOR[11])
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
Multi-Freq-LDPy is a work in progress, and we expect to release new versions frequently, incorporating feedback and code contributions from the community. Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact
For any question, please contact [Héber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr

## Acknowledgments
   * The Local Hashing (LH) functions were adapted from the [pure-LDP](https://github.com/Samuel-Maddock/pure-LDP) package, which covers a wider range of frequency oracles for single-frequency estimation.
   * Some codes were adapted from our [ldp-protocols-mobility-cdrs](https://github.com/hharcolezi/ldp-protocols-mobility-cdrs) repository. 

## License
[MIT](https://github.com/hharcolezi/multi-freq-ldpy/blob/main/LICENSE)


## References
- [1] Kairouz, Peter, Keith Bonawitz, and Daniel Ramage. "Discrete distribution estimation under local privacy." International Conference on Machine Learning. PMLR, 2016.
- [2] Kairouz, Peter, Sewoong Oh, and Pramod Viswanath. "Extremal mechanisms for local differential privacy." Advances in neural information processing systems 27 (2014).
- [3] Wang, Tianhao, et al. "Locally differentially private protocols for frequency estimation." 26th USENIX Security Symposium (USENIX Security 17). 2017.
- [4] Bassily, Raef, and Adam Smith. "Local, private, efficient protocols for succinct histograms." Proceedings of the forty-seventh annual ACM symposium on Theory of computing. 2015.
- [5] Ye, Min, and Alexander Barg. "Optimal schemes for discrete distribution estimation under locally differential privacy." IEEE Transactions on Information Theory 64.8 (2018): 5662-5676.
- [6] Wang, Shaowei, et al. "Mutual information optimally local private discrete distribution estimation." arXiv preprint arXiv:1607.08025 (2016).
- [7] Nguyên, Thông T., et al. "Collecting and analyzing data from smart device users with local differential privacy." arXiv preprint arXiv:1606.05053 (2016).
- [8] Wang, Ning, et al. "Collecting and analyzing multidimensional data with local differential privacy." 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 2019.
- [9] Arcolezi, Héber H., et al. "Random sampling plus fake data: Multidimensional frequency estimates with local differential privacy." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.
- [10] Arcolezi, Héber H., et al. "Improving the utility of locally differentially private protocols for longitudinal and multidimensional frequency estimates." Digital Communications and Networks (2022).
- [11] Erlingsson, Úlfar, Vasyl Pihur, and Aleksandra Korolova. "Rappor: Randomized aggregatable privacy-preserving ordinal response." Proceedings of the 2014 ACM SIGSAC conference on computer and communications security. 2014.
- [12] Ding, Bolin, Janardhan Kulkarni, and Sergey Yekhanin. "Collecting telemetry data privately." Advances in Neural Information Processing Systems 30 (2017).
