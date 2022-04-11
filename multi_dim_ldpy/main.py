
import numpy as np
from pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from pure_frequency_oracles.UE import UE_Client, UE_Aggregator

np.random.seed(42)

print('-----------------------Starting Single Frequency Estimation-----------------------')
# Test dataset
dataset = [0]*40000 + [1]*30000 + [2]*20000 + [3]*10000

k = len(set(dataset))
n = len(dataset)
eps = 1

lst = []

for item in dataset:
    #print(type(item))
    # Simulate client-side privatisation
    lst.append(GRR_Client(item, k, eps))

# Simulate server-side aggregation


print(GRR_Aggregator(lst, k=k, epsilon=eps))

ue = np.eye(k)
ue_dataset = [ue[val] for val in dataset]

lst = []

for item in ue_dataset:
    #print(type(item))
    # Simulate client-side privatisation
    lst.append(UE_Client(item, eps, False))

# Simulate server-side aggregation
print(UE_Aggregator(lst, epsilon=eps, optimal=False))

print('-----------------------Starting Multidimensional-----------------------')
from mult_freq_est.SPL_solution import SPL_GRR_Client, SPL_UE_Client, SPL_GRR_Aggregator, SPL_UE_Aggregator
from mult_freq_est.SMP_solution import SMP_GRR_Client, SMP_UE_Client, SMP_GRR_Aggregator, SMP_UE_Aggregator

eps = 1

lst_k = [10, 2, 5]
input_tuple = [2, 0, 3]
ue10 = np.eye(10)
ue2 = np.eye(2)
ue5 = np.eye(5)
input_ue_tuple = [ue10[input_tuple[0]], ue2[input_tuple[1]], ue5[input_tuple[2]]]
d = len(lst_k)

print('\n-----------------------EXAMPLE SPL-----------------------')
print(SPL_GRR_Client(input_tuple, lst_k, d, eps))
print(SPL_UE_Client(input_ue_tuple, d, eps))

print('-----------------------EXAMPLE SMP-----------------------')
print(SMP_GRR_Client(input_tuple, lst_k, d, eps))
print(SMP_UE_Client(input_ue_tuple, d, eps))

print('-----------------------ESTIMATION SPL-----------------------')
input_dataset = [[np.random.randint(10), np.random.randint(2), np.random.randint(5)] for _ in range(10000)]

lst_spl = [SPL_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]

print(SPL_GRR_Aggregator(lst_spl, lst_k, d, eps))

input_ue_dataset = [[ue10[np.random.randint(10)], ue2[np.random.randint(2)], ue5[np.random.randint(5)]] for _ in range(10000)]

lst_spl_ue = [SPL_UE_Client(input_tuple, d, eps) for input_tuple in input_ue_dataset]

print(SPL_UE_Aggregator(lst_spl_ue, d, eps))

print('-----------------------ESTIMATION SMP-----------------------')
lst_smp = [SMP_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
lst_smp_ue = [SMP_UE_Client(input_tuple, d, eps) for input_tuple in input_ue_dataset]

print(SMP_GRR_Aggregator(lst_smp, lst_k, d, eps))
print(SMP_UE_Aggregator(lst_smp_ue, d, eps))

from mult_freq_est.RSpFD_solution import RSpFD_GRR_Client, RSpFD_GRR_Aggregator
from mult_freq_est.RSpFD_solution import RSpFD_UE_zero_Client, RSpFD_UE_zero_Aggregator
from mult_freq_est.RSpFD_solution import RSpFD_UE_rnd_Client, RSpFD_UE_rnd_Aggregator

lst_rspfd = [RSpFD_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
lst_rspfd_ue_z = [RSpFD_UE_zero_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_ue_dataset]
lst_rspfd_ue_r = [RSpFD_UE_rnd_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_ue_dataset]

print('\n-----------------------EXAMPLE RS+FD-----------------------')
print(lst_rspfd[0])
print(lst_rspfd_ue_z[0])
print(lst_rspfd_ue_r[0])

print('\n-----------------------ESTIMATION RS+FD-----------------------')
print(RSpFD_GRR_Aggregator(lst_rspfd, lst_k, d, eps))
print(RSpFD_UE_zero_Aggregator(lst_rspfd_ue_z, lst_k, d, eps, optimal=True))
print(RSpFD_UE_rnd_Aggregator(lst_rspfd_ue_r, lst_k, d, eps, optimal=True))

print('-----------------------Starting Longitudinal-----------------------')
from long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator

eps_perm = 5
eps_1 = 0.4 * eps_perm

dataset = [0]*4000 + [1]*3000 + [2]*2000 + [3]*1000

k = len(set(dataset))
n = len(dataset)

lst = []

for idx in range(n):
    lst.append(L_GRR_Client(dataset[idx], k, eps_perm, eps_1))

print('-----------------------EXAMPLE L-GRR-----------------------')
print(lst[0])
# Simulate server-side aggregation
print('-----------------------ESTIMATION L-GRR-----------------------')
print(L_GRR_Aggregator(lst, k, eps_perm, eps_1))

from long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator
from long_freq_est.L_OUE import L_OUE_Client, L_OUE_Aggregator
from long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator
from long_freq_est.L_SOUE import L_SOUE_Client, L_SOUE_Aggregator

ue = np.eye(k)
ue_dataset = [ue[val] for val in dataset]

lst_l_oue = []
lst_l_osue = []
lst_l_soue = []
lst_l_sue = []

for idx in range(n):
    lst_l_oue.append(L_OUE_Client(ue_dataset[idx], eps_perm, eps_1))
    lst_l_osue.append(L_OSUE_Client(ue_dataset[idx], eps_perm, eps_1))
    lst_l_soue.append(L_SOUE_Client(ue_dataset[idx], eps_perm, eps_1))
    lst_l_sue.append(L_SUE_Client(ue_dataset[idx], eps_perm, eps_1))

print('-----------------------EXAMPLE L-UE-----------------------')
print(lst_l_oue[0])
print(lst_l_osue[0])
print(lst_l_soue[0])
print(lst_l_sue[0])

print('-----------------------ESTIMATION L-UE-----------------------')
print(L_OUE_Aggregator(lst_l_oue, eps_perm, eps_1))
print(L_OSUE_Aggregator(lst_l_osue, eps_perm, eps_1))
print(L_SOUE_Aggregator(lst_l_soue, eps_perm, eps_1))
print(L_SUE_Aggregator(lst_l_sue, eps_perm, eps_1))



