#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pomegranate import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.mixture import GaussianMixture
import random


# In[61]:


def create_grid(width = 224,height = 224):
    dim = width * height
    edges = []
    flag = 0 
    last_row = width * (height-1)
    for i in range(dim):
        choice = random.randint(0,10)
        if choice >= 9:
          flag = 1
        if flag == 0:
          if i>= last_row:
              edges.append((i+1,))
          elif i%width == 1 and i > 1:
              edges.append((i+width,))
          elif i%width!=0:
              edges.append((i+1,i+width))
        else:
            if i>= last_row:
              edges.append((i+1,50178))
            elif i%width == 1 and i > 1:
              edges.append((i+width,50178))
            elif i%width!=0:
              edges.append((i+1,i+width,50178))
        flag = 0
    edges.append((50178,))
    edges.append((50178,))
    return tuple(edges)


# In[45]:


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


# In[46]:


def extract_features(img):
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
    opening = opening.flatten()
    return opening


# In[47]:


df = pd.read_csv('CheXpert-v1.0-small/train.csv')

columns = ['Pneumothorax','Consolidation','Fracture','Lung Lesion','Enlarged Cardiomediastinum',
            'Pneumonia','Pleural Other','No Finding','Cardiomegaly','Lung Opacity',
            'Edema','Pleural Effusion','Support Devices','AP/PA','Frontal/Lateral']

column = "Atelectasis"
one_df = df[df[column] == 1][:1000]
zero_df = df[df[column] == 0][:1000]
new_df = pd.concat([one_df,zero_df]).sample(frac = 1)
new_df = new_df.drop(columns,axis = 1)
new_df = new_df.dropna()
new_df.head()


# In[48]:


features_list = []
for i in new_df['Path']:
    img = cv2.imread(i,0)
    img = cv2.resize(img,(224,224))
    features_list.append(extract_features(img))


# In[52]:


feature_df = pd.DataFrame(features_list)
new_df['Sex'][new_df['Sex']=='Male'] = 1
new_df['Sex'][new_df['Sex']=='Female'] = 0
feature_df[50176] = new_df.reset_index()['Sex']
feature_df[50177] = new_df.reset_index()['Age']
feature_df[50178] = new_df.reset_index()[column]
feature_df.head()


# In[62]:


edges = create_grid()
print(len(edges))

# In[ ]:


model = MarkovNetwork.from_structure(feature_df.values,structure = edges)


# In[ ]:


with open('markov_'+column.lower()+'_model.pkl','wb') as f:
    pickle.dump(model,f,protocol=4)

