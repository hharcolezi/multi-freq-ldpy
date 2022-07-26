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


# ## Importing Pure Frequency Oracles from multi_freq_ldpy

# In[2]:


from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from multi_freq_ldpy.pure_frequency_oracles.UE import *
from multi_freq_ldpy.pure_frequency_oracles.ADP import *
from multi_freq_ldpy.pure_frequency_oracles.LH import *
from multi_freq_ldpy.pure_frequency_oracles.SS import *


# ## Usage Example

# In[3]:


k = 10 # number of values
input_data = 2 # real input value
eps = 1 # privacy guarantee

print('Real value:', input_data)
print('Sanitization w/ GRR protocol:', GRR_Client(input_data, k, eps)) 
print('Sanitization w/ OUE protocol:', UE_Client(input_data, k, eps, optimal=True))
print('Sanitization w/ OLH protocol:', LH_Client(input_data, eps, optimal=True)) # sanitized value, seed used to hash
print('Sanitization w/ SS protocol:', SS_Client(input_data, k, eps)) # set of sanitized values


# ## Reading Adult dataset with only 'age' attribute

# In[4]:


df = pd.read_csv('datasets/db_adults.csv', usecols=['age'])
df


# ## Encoding values

# In[5]:


LE = LabelEncoder()

df['age'] = LE.fit_transform(df['age'])
df


# ## Static Parameteres

# In[6]:


# number of users (n)
n = df.shape[0]
print('Number of Users =',n)

# attribute's domain size
k = len(set(df['age']))
print("\nAttribute's domain size =", k)

print("\nPrivacy guarantees:")

# range of epsilon
lst_eps = np.arange(0.5, 5.1, 0.5)
print('Epsilon values =', lst_eps)


# ## Comparison of frequency oracles

# In[7]:


# Real normalized frequency
real_freq = np.unique(df, return_counts=True)[-1] / n

# Repeat nb_seed times since DP protocols are randomized
nb_seed = 30

# Save Mean Squared Error (MSE) between real and estimated frequencies per seed
dic_mse = {seed: 
               {
                "GRR": [],
                "SUE": [],
                "OUE": [],
                "BLH": [],
                "OLH": [],
                "SS":  [],
               } 
               for seed in range(nb_seed)
          }

starttime = time.time()
for seed in range(nb_seed):
    print('Starting w/ seed:', seed)

    for eps in lst_eps:
        
        # GRR protocol
        grr_reports = [GRR_Client(input_data, k, eps) for input_data in df['age']]
        grr_est_freq = GRR_Aggregator(grr_reports, k, eps)
        dic_mse[seed]["GRR"].append(mean_squared_error(real_freq, grr_est_freq))

        # SUE protocol
        sue_reports = [UE_Client(input_data, k, eps, optimal=False) for input_data in df['age']]
        sue_est_freq = UE_Aggregator(sue_reports, eps, optimal=False)
        dic_mse[seed]["SUE"].append(mean_squared_error(real_freq, sue_est_freq))

        # OUE protocol
        oue_reports = [UE_Client(input_data, k, eps, optimal=True) for input_data in df['age']]
        oue_est_freq = UE_Aggregator(oue_reports, eps, optimal=True)
        dic_mse[seed]["OUE"].append(mean_squared_error(real_freq, oue_est_freq))

        # BLH protocol        
        blh_reports = [LH_Client(input_data, eps, optimal=False) for input_data in df['age']]
        blh_est_freq = LH_Aggregator(blh_reports, k, eps, optimal=False)
        dic_mse[seed]["BLH"].append(mean_squared_error(real_freq, blh_est_freq))

        # OLH protocol       
        olh_reports = [LH_Client(input_data, eps, optimal=True) for input_data in df['age']]
        olh_est_freq = LH_Aggregator(olh_reports, k, eps, optimal=True)
        dic_mse[seed]["OLH"].append(mean_squared_error(real_freq, olh_est_freq))
        
        # SS protocol       
        ss_reports = [SS_Client(input_data, k, eps) for input_data in df['age']]
        ss_est_freq = SS_Aggregator(ss_reports, k, eps)
        dic_mse[seed]["SS"].append(mean_squared_error(real_freq, ss_est_freq))
        
print('That took {} seconds'.format(time.time() - starttime))        


# ## Plotting metrics results

# In[8]:


plt.figure(figsize=(8,5))
plt.grid(color='grey', linestyle='dashdot', linewidth=0.5)
plt.plot(np.mean([dic_mse[seed]["GRR"] for seed in range(nb_seed)], axis=0), label='GRR', marker='o')
plt.plot(np.mean([dic_mse[seed]["SUE"] for seed in range(nb_seed)], axis=0), label='SUE',marker='>',linestyle='dashed')
plt.plot(np.mean([dic_mse[seed]["OUE"] for seed in range(nb_seed)], axis=0), label='OUE',marker='s',linestyle='dotted')
plt.plot(np.mean([dic_mse[seed]["BLH"] for seed in range(nb_seed)], axis=0), label='BLH', marker='D', linestyle=(0, (3, 10, 1, 10)))
plt.plot(np.mean([dic_mse[seed]["OLH"] for seed in range(nb_seed)], axis=0), label='OLH',marker='d',linestyle=(0, (5, 10)))
plt.plot(np.mean([dic_mse[seed]["SS"] for seed in range(nb_seed)], axis=0), label='SS',marker='X',linestyle=(0, (3, 10, 1, 10)))

plt.yscale('log')
plt.xlabel('$\epsilon$')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(range(len(lst_eps)), lst_eps)
plt.legend(ncol=2)
plt.show()


# ## Example of Real vs Estimated Freqencies

# In[10]:


plt.figure(figsize=(12, 5))

barwidth = 0.4
x_axis = np.arange(k)

plt.bar(x_axis - barwidth, real_freq, label='Real Freq', width=barwidth)
plt.bar(x_axis, ss_est_freq, label='Est Freq: SS', width=barwidth)
plt.ylabel('Normalized Frequency')
plt.xlabel('Age attribute with domain size = {}'.format(k))
plt.legend()
plt.show();


# In[ ]:




