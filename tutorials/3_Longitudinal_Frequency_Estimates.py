#!/usr/bin/env python
# coding: utf-8

# ## Common Libraries

# In[1]:


import matplotlib.pyplot as plt
import matplotlib
params = {'axes.titlesize':'14',
          'xtick.labelsize':'14',
          'ytick.labelsize':'14',
          'font.size':'14',
          'legend.fontsize':'medium',
          'lines.linewidth':'2',
          'font.weight':'normal',
          'lines.markersize':'10'
          }
matplotlib.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('font', family='serif')

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ## Usage Example

# In[22]:


k = 10 # number of values
input_data = 2 # real input value
eps_perm = 2 # epsilon infinity (infinity reports -- upper bound)
eps_1 = 0.5 # epsilon 1 (single report -- lower bound)

print('Real value:', input_data)
print('Sanitization w/ L-GRR protocol:', L_GRR_Client(input_data, k, eps_perm, eps_1)) 
print('Sanitization w/ L-SUE protocol:', L_SUE_Client(input_data, k, eps_perm, eps_1))
print('Sanitization w/ L-OSUE protocol:', L_OSUE_Client(input_data, k, eps_perm, eps_1))


# ## Importing Longitudinal Protocols from multi_freq_ldpy

# In[2]:


from multi_freq_ldpy.long_freq_est.L_GRR import *
from multi_freq_ldpy.long_freq_est.L_OUE import *
from multi_freq_ldpy.long_freq_est.L_OSUE import *
from multi_freq_ldpy.long_freq_est.L_SUE import *
from multi_freq_ldpy.long_freq_est.L_SOUE import *


# ## Reading Adult dataset with only 'age' attribute

# In[3]:


df = pd.read_csv('datasets/db_adults.csv', usecols=['age'])
df


# ## Encoding values

# In[4]:


LE = LabelEncoder()

df['age'] = LE.fit_transform(df['age'])
df


# ## Static Parameteres

# In[5]:


# number of users (n)
n = df.shape[0]
print('Number of Users =',n)

# attribute's domain size
k = len(set(df['age']))
print("\nAttribute's domain size =", k)

print("\nPrivacy guarantees:")

# upper bound (infinity reports)
lst_eps_perm = np.arange(0.5, 5.1, 0.5)
print("List of epsilon_perm =", lst_eps_perm)

#lower bound (single report)
lst_eps_1 = lst_eps_perm * 0.5
print("List of epsilon_1 =", lst_eps_1)


# ## Comparison of longitudinal protocols

# In[6]:


# Real normalized frequency
real_freq = np.unique(df, return_counts=True)[-1] / n

# Repeat nb_seed times since DP protocols are randomized
nb_seed = 30

# Save Mean Squared Error (MSE) between real and estimated frequencies per seed
dic_mse = {seed: 
               {"L_GRR": [],
               "L_OUE": [],
               "L_OSUE": [],
               "L_SUE": [],
               "L_SOUE": [],
               } 
               for seed in range(nb_seed)
          }

starttime = time.time()
for seed in range(nb_seed):
    print('Starting w/ seed:', seed)

    for idx_eps in range(len(lst_eps_perm)):

        eps_perm = lst_eps_perm[idx_eps]
        eps_1 = lst_eps_1[idx_eps]

        # L_GRR protocol
        l_grr_reports = [L_GRR_Client(input_data, k, eps_perm, eps_1) for input_data in df['age']]
        l_grr_est_freq = L_GRR_Aggregator(l_grr_reports, k, eps_perm, eps_1)
        dic_mse[seed]["L_GRR"].append(mean_squared_error(real_freq, l_grr_est_freq))

        # L_OUE protocol
        l_oue_reports = [L_OUE_Client(input_data, k, eps_perm, eps_1) for input_data in df['age']]
        l_oue_est_freq = L_OUE_Aggregator(l_oue_reports, eps_perm, eps_1)
        dic_mse[seed]["L_OUE"].append(mean_squared_error(real_freq, l_oue_est_freq))

        # L_OSUE protocol
        l_osue_reports = [L_OSUE_Client(input_data, k, eps_perm, eps_1) for input_data in df['age']]
        l_osue_est_freq = L_OSUE_Aggregator(l_osue_reports, eps_perm, eps_1)
        dic_mse[seed]["L_OSUE"].append(mean_squared_error(real_freq, l_osue_est_freq))

        # L_SUE protocol
        l_sue_reports = [L_SUE_Client(input_data, k, eps_perm, eps_1) for input_data in df['age']]
        l_sue_est_freq = L_SUE_Aggregator(l_sue_reports, eps_perm, eps_1)
        dic_mse[seed]["L_SUE"].append(mean_squared_error(real_freq, l_sue_est_freq))

        # L_SOUE protocol
        l_soue_reports = [L_SOUE_Client(input_data, k, eps_perm, eps_1) for input_data in df['age']]
        l_soue_est_freq = L_SOUE_Aggregator(l_soue_reports, eps_perm, eps_1)
        dic_mse[seed]["L_SOUE"].append(mean_squared_error(real_freq, l_soue_est_freq))
print('That took {} seconds'.format(time.time() - starttime))        


# ## Plotting metrics results

# In[7]:


plt.figure(figsize=(8,5))
plt.grid(color='grey', linestyle='dashdot', linewidth=0.5)
plt.plot(np.mean([dic_mse[seed]["L_GRR"] for seed in range(nb_seed)], axis=0), label='L_GRR', marker='o')
plt.plot(np.mean([dic_mse[seed]["L_OUE"] for seed in range(nb_seed)], axis=0), label='L_OUE',marker='>',linestyle='dashed')
plt.plot(np.mean([dic_mse[seed]["L_OSUE"] for seed in range(nb_seed)], axis=0), label='L_OSUE',marker='s',linestyle='dotted')
plt.plot(np.mean([dic_mse[seed]["L_SUE"] for seed in range(nb_seed)], axis=0), label='L_SUE', marker='D', linestyle=(0, (3, 10, 1, 10)))
plt.plot(np.mean([dic_mse[seed]["L_SOUE"] for seed in range(nb_seed)], axis=0), label='L_SOUE',marker='d',linestyle=(0, (5, 10)))

plt.yscale('log')
plt.xlabel('$\epsilon_{perm}$')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(range(len(lst_eps_perm)), lst_eps_perm)
plt.legend(ncol=2)
plt.show()


# ## Example of Real vs Estimated Freqencies

# In[8]:


plt.figure(figsize=(12, 5))

barwidth = 0.4
x_axis = np.arange(k)

plt.bar(x_axis - barwidth, real_freq, label='Real Freq', width=barwidth)
plt.bar(x_axis, l_osue_est_freq, label='Est Freq: L_OSUE', width=barwidth)
plt.ylabel('Normalized Frequency')
plt.xlabel('Age attribute with domain size = {}'.format(k))
plt.legend()
plt.show();


# In[ ]:




