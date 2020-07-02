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


def get_accuracy(model,X_test,y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test,pred)


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
columns = ['Pleural Other', 'No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
           'Pneumothorax', 'Pleural Effusion', 'Fracture','Support Devices']

for column in columns:
  filename = "_".join([w for w in column.split(" ")])
  print(column+" Log")
  new_df = pd.read_csv('lpf_'+filename.lower()+'_features.csv')
  new_df = new_df.drop('Path', axis = 1)
  feat_cols = new_df.columns[new_df.columns!='labels']
  X_train,X_test,y_train,y_test = train_test_split(new_df[feat_cols],new_df["labels"],test_size = 0.2,random_state = 43)
  X_train['labels'] = y_train
  list_of_edges = get_model_architecture(X_train)
  model = BayesianModel(list_of_edges)
  model.fit(X_train)
  feature_cols = new_df.columns[new_df.columns!='labels']
  accuracy = get_accuracy(model,X_train[feature_cols],X_train['labels'])
  print("Training Accuracy",accuracy)

  with open('lpf_bayes_'+filename.lower()+'model.pkl','wb') as f:
    pickle.dump(model,f,protocol=4)

  accuracy = get_accuracy(model,X_test[feature_cols],y_test)
  print("Testing Accuracy",accuracy)