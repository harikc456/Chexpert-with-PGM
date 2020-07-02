import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import cv2

def hpf(img,r):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    mask = np.ones((rows,cols,2),np.uint8)
    mask[int(crow-r):int(crow+r), int(ccol-r):int(ccol+r)] = 0
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

df = pd.read_csv('CheXpert-v1.0-small/train.csv')
img_path = df['Path'][df['Frontal/Lateral']=='Frontal']
images = []
for i in img_path:
  img = cv2.imread(i,0)
  img = cv2.resize(img,(224,224))
  hpf_img = hpf(img,1)
  hpf_img.resize(hpf_img.shape[0] * hpf_img.shape[1],1)
  images.append(hpf_img)
  
n_components = 6
gmm = GaussianMixture(n_components)
gmm.fit(images)

with open('gmm_frontal_6.pkl','wb') as f:
  pickle.dump(gmm,f,protocol=4)
  
img_path = df['Path'][df['Frontal/Lateral']=='Lateral']
images = []
for i in img_path:
  img = cv2.imread(i,0)
  img = cv2.resize(img,(224,224))
  hpf_img = hpf(img,1)
  hpf_img.resize(hpf_img.shape[0] * hpf_img.shape[1],1)
  images.append(hpf_img)
  
n_components = 6
gmm = GaussianMixture(n_components)
gmm.fit(images)

with open('gmm_lateral_6.pkl','wb') as f:
  pickle.dump(gmm,f,protocol=4)