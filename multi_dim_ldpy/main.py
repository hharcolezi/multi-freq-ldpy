
import numpy as np
from pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from pure_frequency_oracles.UE import UE_Client, UE_Aggregator

np.random.seed(42)

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
input_ue_tuple = [list(ue10[input_tuple[0]]), list(ue2[input_tuple[1]]), list(ue5[input_tuple[2]])]
d = len(lst_k)

print('-----------------------EXAMPLE SPL-----------------------')
print(SPL_GRR_Client(input_tuple, lst_k, d, eps))
print(SPL_UE_Client(input_ue_tuple, d, eps))

print('-----------------------EXAMPLE SMP-----------------------')
print(SMP_GRR_Client(input_tuple, lst_k, d, eps))
print(SMP_UE_Client(input_ue_tuple, d, eps))

print('-----------------------SPL-----------------------')
input_dataset = [[np.random.randint(10), np.random.randint(2), np.random.randint(5)] for _ in range(10000)]

lst_spl = [SPL_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
print(lst_spl[0])

print(SPL_GRR_Aggregator(lst_spl, lst_k, d, eps))

input_ue_dataset = [[ue10[np.random.randint(10)], ue2[np.random.randint(2)], ue5[np.random.randint(5)]] for _ in range(10000)]

lst_spl_ue = [SPL_UE_Client(input_tuple, d, eps) for input_tuple in input_ue_dataset]
print(lst_spl_ue[0])

print(SPL_UE_Aggregator(lst_spl_ue, d, eps))

print('-----------------------SMP-----------------------')
lst_smp = [SMP_GRR_Client(input_tuple, lst_k, d, eps) for input_tuple in input_dataset]
lst_smp_ue = [SMP_UE_Client(input_tuple, d, eps) for input_tuple in input_ue_dataset]


print(lst_smp[0])
print(lst_smp_ue[0])

print(SMP_GRR_Aggregator(lst_smp, lst_k, d, eps))
print(SMP_UE_Aggregator(lst_smp_ue, d, eps))


print('-----------------------RS+FD-----------------------')





print('-----------------------Starting Longitudinal-----------------------')
from long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator

eps_perm = 3
eps_1 = 0.25 * eps_perm


dataset = [0]*40000 + [1]*30000 + [2]*20000 + [3]*10000

k = len(set(dataset))
n = len(dataset)

lst = []

for item in dataset:
    #print(type(item))
    # Simulate client-side privatisation
    lst.append(L_GRR_Client(item, k, eps_perm, eps_1))

print(lst[0])
# Simulate server-side aggregation
print(L_GRR_Aggregator(lst, k, eps_perm, eps_1))


# ue = np.eye(k)
# ue_dataset = [ue[val] for val in dataset] 
#
# lst = []
#
# for item in ue_dataset:
#     #print(type(item))
#     # Simulate client-side privatisation
#     lst.append(UE_Client(item, eps, False))
    











