#!/usr/bin/env python
# coding: utf-8

# In[18]:


import cv2
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from networkx.algorithms.cycles import find_cycle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture

# In[2]:


df = pd.read_csv('CheXpert-v1.0-small/train.csv')


# In[3]:


model = None


# In[16]:


column = 'Support Devices'
print(column+" Log")
one_df = pd.DataFrame(df['Path'][(df[column]==1) & (df['Frontal/Lateral']=='Frontal')])
zero_df = pd.DataFrame(df['Path'][(df[column]==0) & (df['Frontal/Lateral']=='Frontal')][:len(one_df)])
null_df = pd.DataFrame(df['Path'][(df[column].isna()) & (df['Frontal/Lateral']=='Frontal')][:len(one_df) - len(zero_df)])
zero_df['labels'] = 0
one_df['labels'] = 1
null_df['labels'] = 0
new_df = pd.concat([one_df,zero_df,null_df]).sample(frac = 1)
new_df['Sex'] = df['Sex']
new_df['Sex'][new_df['Sex']=='Male'] = 1
new_df['Sex'][new_df['Sex']=='Female'] = 0
new_df['Sex'][new_df['Sex']=='Unknown'] = 0
new_df['Sex'] = new_df["Sex"].astype(int)
new_df['Age'] = df['Age']
new_df['Age'][new_df['Age'] < 30] = 1
new_df['Age'][new_df['Age'] >= 30] = 2
new_df['Age'][new_df['Age'] > 60] = 3
new_df['Age'] = new_df["Age"].astype(int)

# In[5]:


class Feature_network(nn.Module):
    def __init__(self):
        super(Feature_network,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(1,5,kernel_size = (11,11))
        self.fc = nn.Linear(228980 , 64)
          
    def forward(self,x):
        t = self.conv(x)
        t = t.view(1,-1)
        t = self.fc(t)
        t = self.sigmoid(t)
        return t


# In[6]:


def hpf(img,r):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows,cols,2),np.uint8)
    mask[int(crow-r):int(crow+r), int(ccol-r):int(ccol+r)] = 0
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back


# In[7]:


def extract_features(img,net):
    features = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    hpf_img = hpf(img,1)
    shape = hpf_img.shape
    hpf_img.resize(hpf_img.shape[0] * hpf_img.shape[1],1)
    n_components = 3
    gmm = GaussianMixture(n_components,max_iter=250)
    gmm.fit(hpf_img)
    pred_img = gmm.predict(hpf_img)
    pred_img.resize(shape)
    mask = np.zeros((img.shape[0],img.shape[1]))
    mask[pred_img != 0] = 1
    segment_img = mask * img
    opening = cv2.morphologyEx(segment_img, cv2.MORPH_OPEN, kernel)
    x = torch.tensor(opening, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
    y = feature_net(x).squeeze(0)
    y[y>0.5] = 1
    y[y<=0.5] = 0
    return y.tolist()


# In[8]:


def get_accuracy(model,X_test,y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test,pred)


# In[9]:


def get_loss(model,criterion,X_test,y_test):
    pred_prob = model.predict_probability(X_test)
    pred_prob = torch.tensor(pred_prob.values,requires_grad = True)
    y_ = torch.tensor(y_test.values, dtype=torch.int64)
    return criterion(pred_prob,y_)


# In[10]:


def check_dag(edge_list):
    G = nx.DiGraph(edge_list)
    try:
        cycle = find_cycle(G, orientation='original')
    except:
        return False
    return cycle


# In[11]:


def generate_edges(child,parents):
    edge_list = []
    for parent in parents:
        edge_list.append([parent,child])
    return edge_list


# In[12]:


def get_cycle_edge(cycle,score,list_of_edges):
    cycle_edge_scores = []
    for parent,child,direction in cycle:
        cycle_edge_scores.append(score.local_score(child,[parent]))
    min_index = cycle_edge_scores.index(min(cycle_edge_scores))
    low_score_edge = cycle[min_index][:-1]
    return low_score_edge


# In[13]:


def get_model_architecture(df):
    score = K2Score(df)
    list_of_edges = []
    for i in df.columns:
        edge_scores = []
        for j in df.columns:
            if i!=j:
                sco =score.local_score(i,[j])
                edge_scores.append((i,j,sco))
        edge_scores.sort(key = lambda x:x[2],reverse = True)
        parents = [edge_scores[0][1]]
        best_score = edge_scores[0][2]
        for v in range(1,10):
            parents.append(edge_scores[v][1])
            new_score = score.local_score(i,parents)
            if new_score > best_score:
                best_score = new_score
            else:
                parents = parents[:-1]
                break
        list_of_edges += generate_edges(i,parents)
        cycle = check_dag(list_of_edges)
        while cycle:
            low_score_edge = get_cycle_edge(cycle,score,list_of_edges)
            list_of_edges.remove(list(low_score_edge))
            cycle = check_dag(list_of_edges)
    return list_of_edges


# In[14]:

feat_cols = new_df.columns[new_df.columns!=column]
X_train,X_test,y_train,y_test = train_test_split(new_df[feat_cols],new_df["labels"],test_size = 0.2,random_state = 43)
X_train['labels'] = y_train
batch_size = 500
iterations = 1
feature_net = Feature_network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(feature_net.parameters(), lr=0.005)
n_components = 3
'''
for itera in range(iterations):
    start_time = time.time()
    features_list = []
    paths = X_train.sample(batch_size)
    for i in paths['Path']:
        img = cv2.imread(i,0)
        img = cv2.resize(img,(224,224))
        features_list.append(extract_features(img,feature_net))
    feature_df = pd.DataFrame(features_list)
    col_names = [str(c) for c in feature_df.columns]
    feature_df.columns = col_names
    feature_df['Sex'] = paths.reset_index()['Sex']
    feature_df['Age'] = paths.reset_index()['Age']
    feature_df['labels'] = paths.reset_index()['labels']
    list_of_edges = get_model_architecture(feature_df)
    model = BayesianModel(list_of_edges)
    model.fit(feature_df)
    feature_cols = feature_df.columns[feature_df.columns!='labels']
    accuracy = get_accuracy(model,feature_df[feature_cols],feature_df['labels'])
    loss = get_loss(model,criterion,feature_df[feature_cols],feature_df['labels'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end_time = time.time()
    batch_time = end_time - start_time
    print("Iteration: {} Accuracy: {} Loss: {} ----- {}s ".format(itera,accuracy,loss.item(),batch_time))
'''

# In[19]:


torch.save(feature_net,column.lower()+"_net.pt")


# In[ ]:

for i in X_train['Path']:
    img = cv2.imread(i,0)
    img = cv2.resize(img,(224,224))
    features_list.append(extract_features(img,feature_net))
feature_df = pd.DataFrame(features_list)
col_names = [str(c) for c in feature_df.columns]
feature_df.columns = col_names
feature_df['Sex'] = X_train['Sex']
feature_df['Age'] = X_train['Age']
feature_df['labels'] = X_train['labels']
feature_df = feature_df.dropna()
list_of_edges = get_model_architecture(feature_df)
model = BayesianModel(list_of_edges)
model.fit(feature_df)
accuracy = get_accuracy(model,feature_df[feature_cols],feature_df['labels'])
print("Training Accuracy",accuracy)

# In[ ]:

feature_df.to_csv(column.lower()+"_features.csv")
with open('bayes_'+column.lower()+'model.pkl','wb') as f:
  pickle.dump(model,f,protocol=4)

# In[ ]:
features_list = []
for i in X_test['Path']:
    img = cv2.imread(i,0)
    img = cv2.resize(img,(224,224))
    features_list.append(extract_features(img,feature_net))
feature_df = pd.DataFrame(features_list)
col_names = [str(c) for c in feature_df.columns]
feature_df.columns = col_names
feature_df['Sex'] = X_test['Sex']
feature_df['Age'] = X_test['Age']
feature_df['labels'] = y_test
feature_df = feature_df.dropna()
accuracy = get_accuracy(model,feature_df[feature_cols],feature_df['labels'])
print("Validation Accuracy",accuracy)


