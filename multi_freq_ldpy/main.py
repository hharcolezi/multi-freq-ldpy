
import numpy as np
from pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from pure_frequency_oracles.UE import UE_Client, UE_Aggregator
from pure_frequency_oracles.ADP import ADP_Client, ADP_Aggregator

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

lst = []

for item in dataset:
    #print(type(item))
    # Simulate client-side privatisation
    lst.append(UE_Client(item, k, eps, False))

# Simulate server-side aggregation
print(UE_Aggregator(lst, epsilon=eps, optimal=False))

lst = []

for item in dataset:
    #print(type(item))
    # Simulate client-side privatisation
    lst.append(ADP_Client(item, k, eps, False))

# Simulate server-side aggregation
print(ADP_Aggregator(lst, k, epsilon=eps, optimal=False))

print('-----------------------Starting Multidimensional-----------------------')
from mdim_freq_est.SPL_solution import SPL_GRR_Client, SPL_UE_Client, SPL_GRR_Aggregator, SPL_UE_Aggregator, \
    SPL_ADP_Client, SPL_ADP_Aggregator
from mdim_freq_est.SMP_solution import SMP_GRR_Client, SMP_UE_Client, SMP_GRR_Aggregator, SMP_UE_Aggregator, \
    SMP_ADP_Client, SMP_ADP_Aggregator

eps = 3

lst_k = [2, 3, 15]
input_tuple = [2, 0, 1]
d = len(lst_k)

print('\n-----------------------EXAMPLE SPL-----------------------')
print(SPL_GRR_Client(input_tuple, lst_k, d, eps))
print(SPL_UE_Client(input_tuple, lst_k, d, eps))
print(SPL_ADP_Client(input_tuple, lst_k, d, eps, optimal=True))

print('-----------------------EXAMPLE SMP-----------------------')
print(SMP_GRR_Client(input_tuple, lst_k, d, eps))
print(SMP_UE_Client(input_tuple, lst_k, d, eps))
print(SMP_ADP_Client(input_tuple, lst_k, d, eps, optimal=True))

print('-----------------------ESTIMATION SPL-----------------------')
input_dataset = [[np.random.randint(lst_k[0]), np.random.randint(lst_k[1]), np.random.randint(lst_k[2])] for _ in range(10000)]

lst_spl = [SPL_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]

print(SPL_GRR_Aggregator(lst_spl, lst_k, d, eps))

lst_spl_ue = [SPL_UE_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]

print(SPL_UE_Aggregator(lst_spl_ue, d, eps))

lst_spl_adp = [SPL_ADP_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_dataset]

print(SPL_ADP_Aggregator(lst_spl_adp, lst_k, d, eps, optimal=True))

print('-----------------------ESTIMATION SMP-----------------------')
lst_smp = [SMP_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
lst_smp_ue = [SMP_UE_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
lst_smp_adp = [SMP_ADP_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_dataset]

print(SMP_GRR_Aggregator(lst_smp, lst_k, d, eps))
print(SMP_UE_Aggregator(lst_smp_ue, d, eps))
print(SMP_ADP_Aggregator(lst_smp_adp, lst_k, d, eps, optimal=True))

from mdim_freq_est.RSpFD_solution import RSpFD_GRR_Client, RSpFD_GRR_Aggregator
from mdim_freq_est.RSpFD_solution import RSpFD_UE_zero_Client, RSpFD_UE_zero_Aggregator
from mdim_freq_est.RSpFD_solution import RSpFD_UE_rnd_Client, RSpFD_UE_rnd_Aggregator
from mdim_freq_est.RSpFD_solution import RSpFD_ADP_Client, RSpFD_ADP_Aggregator

lst_rspfd = [RSpFD_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
lst_rspfd_ue_z = [RSpFD_UE_zero_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_dataset]
lst_rspfd_ue_r = [RSpFD_UE_rnd_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_dataset]
lst_rspfd_adp = [RSpFD_ADP_Client(input_tuple, lst_k, d, eps, optimal=True) for input_tuple in input_dataset]

print('\n-----------------------EXAMPLE RS+FD-----------------------')
print(lst_rspfd[0])
print(lst_rspfd_ue_z[0])
print(lst_rspfd_ue_r[0])
print(lst_rspfd_adp[0])

print('\n-----------------------ESTIMATION RS+FD-----------------------')
print(RSpFD_GRR_Aggregator(lst_rspfd, lst_k, d, eps))
print(RSpFD_UE_zero_Aggregator(lst_rspfd_ue_z, lst_k, d, eps, optimal=True))
print(RSpFD_UE_rnd_Aggregator(lst_rspfd_ue_r, lst_k, d, eps, optimal=True))
print(RSpFD_ADP_Aggregator(lst_rspfd_adp, lst_k, d, eps, optimal=True))

print('-----------------------Starting Longitudinal-----------------------')
from long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator

eps_perm = 3
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
from long_freq_est.L_ADP import L_ADP_Client, L_ADP_Aggregator

lst_l_oue = []
lst_l_osue = []
lst_l_soue = []
lst_l_sue = []
lst_l_adp = []

for idx in range(n):
    lst_l_oue.append(L_OUE_Client(dataset[idx], k, eps_perm, eps_1))
    lst_l_osue.append(L_OSUE_Client(dataset[idx], k, eps_perm, eps_1))
    lst_l_soue.append(L_SOUE_Client(dataset[idx], k, eps_perm, eps_1))
    lst_l_sue.append(L_SUE_Client(dataset[idx], k, eps_perm, eps_1))
    lst_l_adp.append(L_ADP_Client(dataset[idx], k, eps_perm, eps_1))

print('-----------------------EXAMPLE L-UE-----------------------')
print(lst_l_oue[0])
print(lst_l_osue[0])
print(lst_l_soue[0])
print(lst_l_sue[0])
print(lst_l_adp[0])

print('-----------------------ESTIMATION L-UE-----------------------')
print(L_OUE_Aggregator(lst_l_oue, eps_perm, eps_1))
print(L_OSUE_Aggregator(lst_l_osue, eps_perm, eps_1))
print(L_SOUE_Aggregator(lst_l_soue, eps_perm, eps_1))
print(L_SUE_Aggregator(lst_l_sue, eps_perm, eps_1))
print(L_ADP_Aggregator(lst_l_adp, k, eps_perm, eps_1))


print('-----------------------Starting Longitudinal and Multidimensional-----------------------')
from long_mdim_freq_est.L_SPL_Solution import SPL_L_GRR_Client, SPL_L_OUE_Client, SPL_L_OSUE_Client, SPL_L_SUE_Client, \
    SPL_L_SOUE_Client, SPL_L_ADP_Client
from long_mdim_freq_est.L_SPL_Solution import SPL_L_GRR_Aggregator, SPL_L_OUE_Aggregator, SPL_L_OSUE_Aggregator, \
    SPL_L_SUE_Aggregator, SPL_L_SOUE_Aggregator, SPL_L_ADP_Aggregator
from long_mdim_freq_est.L_SMP_Solution import  SMP_L_GRR_Client, SMP_L_OUE_Client, SMP_L_OSUE_Client, SMP_L_SUE_Client, \
    SMP_L_SOUE_Client, SMP_L_ADP_Client
from long_mdim_freq_est.L_SMP_Solution import SMP_L_GRR_Aggregator, SMP_L_OUE_Aggregator, SMP_L_OSUE_Aggregator, \
    SMP_L_SUE_Aggregator, SMP_L_SOUE_Aggregator, SMP_L_ADP_Aggregator

lst_k = [2, 3, 15]
d = len(lst_k)

eps_perm = 2
eps_1 = 0.4 * eps_perm

input_dataset = [[np.random.randint(lst_k[0]), np.random.randint(lst_k[1]), np.random.randint(lst_k[2])] for _ in range(10000)]

lst_spl = [SPL_L_GRR_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_spl_oue = [SPL_L_OUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_spl_osue = [SPL_L_OSUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_spl_sue = [SPL_L_SUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_spl_soue = [SPL_L_SOUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_spl_adp = [SPL_L_ADP_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]

print('-----------------------EXAMPLE SPL_L-----------------------')
print(lst_spl[0])
print(lst_spl_oue[0])
print(lst_spl_osue[0])
print(lst_spl_sue[0])
print(lst_spl_soue[0])
print(lst_spl_adp[0])

print('-----------------------ESTIMATION SPL_L-----------------------')
print(SPL_L_GRR_Aggregator(lst_spl, lst_k, d, eps_perm, eps_1))
print(SPL_L_OUE_Aggregator(lst_spl_oue, d, eps_perm, eps_1))
print(SPL_L_OSUE_Aggregator(lst_spl_osue, d, eps_perm, eps_1))
print(SPL_L_SUE_Aggregator(lst_spl_sue, d, eps_perm, eps_1))
print(SPL_L_SOUE_Aggregator(lst_spl_soue, d, eps_perm, eps_1))
print(SPL_L_ADP_Aggregator(lst_spl_adp, lst_k, d, eps_perm, eps_1))

lst_smp = [SMP_L_GRR_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_smp_oue = [SMP_L_OUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_smp_osue = [SMP_L_OSUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_smp_sue = [SMP_L_SUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_smp_soue = [SMP_L_SOUE_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]
lst_smp_adp = [SMP_L_ADP_Client(input_tuple, lst_k, d, eps_perm, eps_1) for input_tuple in input_dataset]

print('-----------------------EXAMPLE SMP_L-----------------------')
print(lst_smp[0])
print(lst_smp_oue[0])
print(lst_smp_osue[0])
print(lst_smp_sue[0])
print(lst_smp_soue[0])
print(lst_smp_adp[0])

print('-----------------------ESTIMATION SMP_L-----------------------')
print(SMP_L_GRR_Aggregator(lst_smp, lst_k, d, eps_perm, eps_1))
print(SMP_L_OUE_Aggregator(lst_smp_oue, d, eps_perm, eps_1))
print(SMP_L_OSUE_Aggregator(lst_smp_osue, d, eps_perm, eps_1))
print(SMP_L_SUE_Aggregator(lst_smp_sue, d, eps_perm, eps_1))
print(SMP_L_SOUE_Aggregator(lst_smp_soue, d, eps_perm, eps_1))
print(SMP_L_ADP_Aggregator(lst_smp_adp, lst_k, d, eps_perm, eps_1))




