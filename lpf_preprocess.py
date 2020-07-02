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

df = pd.read_csv('CheXpert-v1.0-small/train.csv')


columns = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
           'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture','Support Devices']

string_cols = ['Sex', 'Age']

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


def lpf(img,r):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[int(crow-r):int(crow+r), int(ccol-r):int(ccol+r)] = 1
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

def extract_features(img,net):
    features = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    hpf_img = lpf(img,100)
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
    
feature_net = torch.load("feature_net.pt")
feature_net.eval()
features_list = []
feature_net = torch.load("feature_net.pt")
feature_net.eval()
df['Sex'][df['Sex']=='Male'] = 1
df['Sex'][df['Sex']=='Female'] = 0
df['Sex'][df['Sex']=='Unknown'] = 0
df['Sex'] = df["Sex"].astype(int)
df['Age'] = df['Age']//5
df['Age'] = df["Age"].astype(int)
for i in df['Path']:
    img = cv2.imread(i,0)
    img = cv2.resize(img,(224,224))
    img = 255 - img
    features_list.append(extract_features(img,feature_net))
feature_df = pd.DataFrame(features_list)
feature_df.to_csv('features_lpf.csv',index = False)

lpf_df = pd.concat([df[columns+string_cols],feature_df],axis = 1)
lpf_df.to_csv("lpf_preprocessed_features.csv", index = False)
