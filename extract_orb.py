import cv2
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('CheXpert-v1.0-small/train.csv')
img_path = df['Path']
images = []
orb = cv2.ORB_create()
for i in img_path:
  img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
  kp = orb.detect(img,None)
  kp,des = orb.compute(img,kp)
  features = []
  for k in kp:
    h,w = k.pt
    features.append(np.mean(img[int(w)][int(h)]))
  images.append(features)
  
images = np.array(images)
print(images.shape)
with open('gray_features.pkl','wb') as f:
  pickle.dump(images,f,protocol=4)