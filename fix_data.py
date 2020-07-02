#!/usr/bin/env python
# coding: utf-8

# In[128]:


import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BdeuScore
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import time
import pickle
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx


# In[38]:


df = pd.read_csv('CheXpert-v1.0-small/train.csv')
df.head()


# In[39]:


df['Sex'] = df['Sex'].fillna(-1)
df['Age'] = df['Age'].fillna(-1)
df['Sex'][df['Sex']=='Male'] = 1
df['Sex'][df['Sex']=='Female'] = 0
df['Sex'][df['Sex']=='Unknown'] = -1
df['Sex'] = df["Sex"].astype(int)
df.head()


# In[40]:


df.isna().sum()


# In[41]:


t = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity', 'Lung Lesion', 'Edema','Consolidation',
     'Pneumonia','Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']
for i,r in df.iterrows():
    if r['No Finding']==1.0:
        r[t].fillna(0)
        for j in t:
            df.at[i,j] = 0


# In[42]:


df['No Finding'] = df['No Finding'].fillna(0)
df.head()


# In[45]:


df['Age'] = df['Age']//5
max(df['Age']), min(df['Age'])


# In[48]:


train_df = df.copy()
t = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity', 'Lung Lesion', 'Edema','Consolidation',
     'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']
features = ['Sex','Age','No Finding'] + t
for i in t:
    train_df[i] = train_df[i].fillna(-1)
train_df = train_df[features]
train_df.head()



with open("fillna_bayesian_network.pkl",'rb') as f:
    model = pickle.load(f)


# In[127]:


new = []
for i,r in df[features].iterrows():
    drop_indices = train_df.columns[list(r[features].isna())]
    if len(drop_indices):
        m = r.drop(drop_indices)
        m = pd.DataFrame(list([m]), columns = m.index.tolist())
        p = model.predict(m)
        for col in p.columns:
            r[col] = p[col]
    new.append(r)


# In[ ]:


new_df = pd.DataFrame(new)
new_df.to_csv('fixed_train.csv',index = False)


# In[ ]:




