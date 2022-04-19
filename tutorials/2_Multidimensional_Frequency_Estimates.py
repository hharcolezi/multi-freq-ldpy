#!/usr/bin/env python
# coding: utf-8

# ## Common Libraries

# In[8]:


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


# ## Importing Multidimensional Protocols from multi_freq_ldpy

# In[2]:


from multi_freq_ldpy.mdim_freq_est.RSpFD_solution import *
from multi_freq_ldpy.mdim_freq_est.SMP_solution import *
from multi_freq_ldpy.mdim_freq_est.SPL_solution import *


# ## Reading MS-FIMU dataset

# In[3]:


df = pd.read_csv('datasets/db_ms_fimu.csv')
df


# ## Encoding values

# In[4]:


LE = LabelEncoder()

attributes = df.columns

for col in attributes:

    df[col] = LE.fit_transform(df[col])
df


# ## Static Parameteres

# In[5]:


# number of users
n = df.shape[0]
print('Number of Users =',n)

# number of attributes
d = len(attributes)
print('Number of Attributes =', d)

# domain size of attributes
lst_k = [len(df[att].unique()) for att in attributes]
print('Domain size of attributes =', lst_k)

print("\nPrivacy guarantees:")

# range of epsilon
lst_eps = np.arange(0.5, 5.1, 0.5)
print('Epsilon values =', lst_eps)


# ## Comparison of multidimensional solutions with single-time protocols

# In[6]:


# Real normalized frequencies
real_freq = [np.unique(df[att], return_counts=True)[-1] / n for att in attributes]

# Repeat nb_seed times since DP protocols are randomized
nb_seed = 50

# Save Averaged Mean Squared Error (MSE_avg) between real and estimated frequencies per seed
dic_avg_mse = {seed: 
                   {"SPL_ADP": [],
                   "SMP_ADP": [],
                   "RSpFD_ADP": []
                   } 
                   for seed in range(nb_seed)
              }

starttime = time.time()
for seed in range(nb_seed):
    print('Starting w/ seed:', seed)

    for eps in lst_eps:
        
        # SPL solution
        spl_reports = [SPL_ADP_Client(input_data, lst_k, d, eps) for input_data in df.values]
        spl_est_freq = SPL_ADP_Aggregator(spl_reports, lst_k, d, eps)
        dic_avg_mse[seed]["SPL_ADP"].append(np.mean([mean_squared_error(real_freq[att], spl_est_freq[att]) for att in range(d)]))

        # SMP solution
        smp_reports = [SMP_ADP_Client(input_data, lst_k, d, eps) for input_data in df.values]
        smp_est_freq = SMP_ADP_Aggregator(smp_reports, lst_k, d, eps)
        dic_avg_mse[seed]["SMP_ADP"].append(np.mean([mean_squared_error(real_freq[att], smp_est_freq[att]) for att in range(d)]))
        
        # RSpFD solution
        rspfd_reports = [RSpFD_ADP_Client(input_data, lst_k, d, eps) for input_data in df.values]
        rspfd_est_freq = RSpFD_ADP_Aggregator(rspfd_reports, lst_k, d, eps)
        dic_avg_mse[seed]["RSpFD_ADP"].append(np.mean([mean_squared_error(real_freq[att], rspfd_est_freq[att]) for att in range(d)]))
print('That took {} seconds'.format(time.time() - starttime))        


# ## Plotting metrics results

# In[22]:


plt.figure(figsize=(8,5))
plt.grid(color='grey', linestyle='dashdot', linewidth=0.5)
plt.plot(np.mean([dic_avg_mse[seed]["SPL_ADP"] for seed in range(nb_seed)], axis=0), label='SPL_ADP', marker='o')
plt.plot(np.mean([dic_avg_mse[seed]["SMP_ADP"] for seed in range(nb_seed)], axis=0), label='SMP_ADP',marker='>',linestyle='dashed')
plt.plot(np.mean([dic_avg_mse[seed]["RSpFD_ADP"] for seed in range(nb_seed)], axis=0), label='RSpFD_ADP',marker='s',linestyle='dotted')

plt.yscale('log')
plt.xlabel('$\epsilon$')
plt.ylabel('$MSE_{avg}$')
plt.xticks(range(len(lst_eps)), lst_eps)
plt.legend(ncol=1)
plt.show();


# ## Example of Real vs Estimated Freqencies

# In[23]:


plt.figure(figsize=(12, 5))

barwidth = 0.4
x_axis = np.arange(sum(lst_k))

plt.bar(x_axis - barwidth, np.concatenate(real_freq), label='Real Freq', width=barwidth)
plt.bar(x_axis, np.concatenate(rspfd_est_freq), label='Est Freq: RspFD_ADP', width=barwidth)
plt.ylabel('Normalized Frequency')
plt.xlabel('d = {} attributes with domain size = {}'.format(d, lst_k))
plt.legend()
plt.show();


# In[ ]:




