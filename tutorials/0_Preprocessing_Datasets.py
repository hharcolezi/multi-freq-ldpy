#!/usr/bin/env python
# coding: utf-8

# # Main libraries

# In[1]:


import pandas as pd
import numpy as np


# # MS-FIMU dataset

# ## Reading dataset (https://github.com/hharcolezi/OpenMSFIMU)

# In[15]:


url = "https://raw.githubusercontent.com/hharcolezi/OpenMSFIMU/master/Data/Personal_table.csv"

df = pd.read_csv(url, error_bad_lines=False)
df.reset_index(inplace=True, drop=True)
df


# ## Dropping columns

# In[16]:


# Non-used columns

cols_to_drop = ["Person ID", "Name"]


# ## Final dataset to work with

# In[17]:


finaldf = df.drop(cols_to_drop, axis=1)
finaldf.to_csv('datasets/db_ms_fimu.csv',index=False)
finaldf


# # Adult dataset

# ## Reading data (https://archive.ics.uci.edu/ml/datasets/adult)

# In[2]:


col = ['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country','salary']
df1 = pd.read_csv('datasets/adult.data' ,header=None)
df2 = pd.read_csv('datasets/adult.test', header=None)
df1.columns = col
df2.columns = col
df = pd.concat([df1,df2])
df


# ## Detecting Missing Values

# In[3]:


df.isin([' ?']).sum(axis=0)


# ## Detecting difference on set of values

# In[4]:


set(df['salary'])


# ## Cleaning missing data / fixing set of values

# In[5]:


df['native-country'] = df['native-country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)
df['salary'].replace(' <=50K','<=50K',inplace=True)
df['salary'].replace(' >50K','>50K',inplace=True)
df['salary'].replace(' <=50K.','<=50K',inplace=True)
df['salary'].replace(' >50K.','>50K',inplace=True)

#dropping the NaN rows now 
df.dropna(how='any',inplace=True)


# ## Checking attributes

# In[6]:


for col in df.columns:
    val_att = set(df[col])
    if len(val_att) < 1000:
        print(col, set(df[col]), len(set(df[col])))
    else:
        print(col, len(set(df[col])))
    print()


# ## Dropping columns

# In[7]:


# education-num is equal to education
# fnlwgt has too many values

cols_to_drop = ['fnlwgt','education-num']


# ## Final dataset to work with

# In[8]:


finaldf = df.drop(cols_to_drop, axis=1)
finaldf.to_csv('datasets/db_adults.csv',index=False)
finaldf


# # Nursery dataset

# ## Reading data (https://archive.ics.uci.edu/ml/datasets/nursery)

# In[13]:


columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "NURSERY"]

df = pd.read_csv('datasets/nursery.data', header=None)
df.columns = columns
df


# ## Final dataset to work with

# In[14]:


df.to_csv('datasets/db_nursery.csv',index=False)
df


# In[ ]:




