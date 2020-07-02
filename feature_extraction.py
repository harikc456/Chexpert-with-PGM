import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import time
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
from pgmpy.factors.discrete import TabularCPD
from collections import Counter

df = pd.read_csv('CheXpert-v1.0-small/train.csv')

class Feature_network(nn.Module):
    def __init__(self):
        super(Feature_network,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(1,5,kernel_size = (11,11))
        self.fc = nn.Linear(228980 , 100)
        
        
    def forward(self,x):
        t = self.conv(x)
        t = t.view(1,-1)
        t = self.fc(t)
        t = self.sigmoid(t)
        return t
        
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
    
def extract_features(img,net,n_components):
    features = []
    hpf_img = hpf(img,1)
    shape = hpf_img.shape
    hpf_img.resize(hpf_img.shape[0] * hpf_img.shape[1],1)
    n_components = 3
    gmm = GaussianMixture(n_components)
    gmm.fit(hpf_img)
    pred_img = gmm.predict(hpf_img)
    pred_img.resize(shape)
    for i in range(n_components):
        mask = np.zeros((img.shape[0],img.shape[1]))
        cluster_number = i
        mask[pred_img == cluster_number] = 1
        segment_img = mask * img
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        opening = cv2.morphologyEx(segment_img, cv2.MORPH_OPEN, kernel)
        x = torch.tensor(opening, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
        y = feature_net(x).squeeze(0)
        y[y>0.5] = 1
        y[y<=0.5] = 0
        features.extend(y.tolist())
    return features

column = 'Pneumonia'
one_df = pd.DataFrame(df['Path'][(df[column]==1) & (df['Frontal/Lateral']=='Frontal')])
zero_df = pd.DataFrame(df['Path'][(df[column]==0) & (df['Frontal/Lateral']=='Frontal')][:len(one_df)])
null_df = pd.DataFrame(df['Path'][(df[column].isna()) & (df['Frontal/Lateral']=='Frontal')][:len(one_df) - len(zero_df)])
zero_df['labels'] = 0
one_df['labels'] = 1
df = pd.concat([one_df,zero_df,null_df]).sample(frac = 1)

s = time.time()
features_list = []
feature_net = Feature_network()
n_components = 3
for n,i in enumerate(df['Path']):
    if n%1000==0:
        print(n)
    img = cv2.imread(i,0)
    img = cv2.resize(img,(224,224))
    features_list.append(extract_features(img,feature_net,n_components))
    
with open('features_list_'+label+'.pkl','wb') as f:
  pickle.dump(features_list,f,protocol=4)
print(time.time() - s)

feature_df = pd.DataFrame(features_list)
feature_df['labels'] = df['labels']
feature_df.to_csv('features_'+label+'.csv',index = False)
torch.save(feature_net,'feature_net_'+label+'.pt')
est = HillClimbSearch(feature_df, scoring_method=BicScore(df))
best_model = est.estimate()

edges = best_model.edges()
model = BayesianModel(edges)