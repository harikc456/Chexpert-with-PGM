#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[2]:


df = pd.read_csv('CheXpert-v1.0-small/train.csv')
df.head()


# In[3]:


orb = cv2.ORB_create()
base_path = 'F:\\'


# In[15]:


encoded_images = []
labels = []
for j,i in df.iterrows():
    img = cv2.imread(i['Path'],0)
    kp = orb.detect(img,None)
    kp,des = orb.compute(img,kp)
    encoded = []
    for point in kp:
        w,h = point.pt
        encoded.append(1 if np.mean(img[int(h)][int(w)])> 127 else 0 )
    encoded_images.append(encoded)
    
mini = min([len(i) for i in encoded_images])

min_encoded_images = []
for i in encoded_images:
    min_encoded_images.append(i[:100])

# In[17]:


orb_features = np.array(min_encoded_images)


# In[27]:


with open('orb_features.pkl','wb') as f:
    pickle.dump(orb_features,f)


# In[ ]:




