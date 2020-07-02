from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pandas as pd
from networkx.drawing.nx_pylab import draw_networkx
from networkx.algorithms.cycles import find_cycle
import random
from sklearn.metrics import accuracy_score
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import pickle

df = pd.read_csv('valid.csv')

def regex_path(path):
    if 'study1' not in path:
        return False
    return True
    
path= df['Path'].apply(regex_path)
study1_df = df[path]

study1_frontal_df = study1_df[study1_df['Frontal/Lateral'] == 'Frontal']
extra_feartures = ['Age','Sex','No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                    'Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis',
                    'Pneumothorax','Pleural Other','Fracture','Support Devices']

features = study1_frontal_df[extra_feartures].fillna(0)
img_size = 224 * 224 *1

model = load_model('conv_autoencoder.h5')

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('dense').output)
intermediate_layer_model.summary()

encoded_images = []
labels = []
for j,i in study1_frontal_df.iterrows():
    img = load_img(i['Path'], grayscale = True, target_size=(224,224))
    img = img_to_array(img)
    img.resize(1,224,224,1)
    encoded = intermediate_layer_model.predict(img, steps = None)
    if j%1000 == 0:
        print(j)
    encoded_images.append(encoded)
    
encoded_images = np.array(encoded_images)

encoded_images.shape = (encoded_images.shape[0], 128)

column_size = 128
column_prefix = 'int_'
columns = []
for i in range(1,column_size+1):
    columns.append(column_prefix+str(i))
    
df = pd.DataFrame(encoded_images, columns = columns)

with open('bayes_model_pgmpy_2(243).pkl','rb') as f:
  model = pickle.load(f)
  
index = [i for i in range(encoded_images.shape[0])]
features['index']=index
features = features.set_index('index')

final_df = pd.concat([df.reindex(),features.reindex()],axis = 1, ignore_index=True)
final_df.columns = columns + extra_feartures
final_df.to_csv('conv_encoded_valid_images_128.csv',index = False)

df = pd.read_csv('conv_encoded_valid_images_128.csv')

label_cols = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
              'Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis',
              'Pneumothorax','Pleural Other','Fracture','Support Devices']

test_df = df[list(model.nodes())]
#for col in label_cols:
#    pred = []
#    for i in range(len(test_df)):
#        test = pd.DataFrame(test_df.loc[i])
#        test = test.T
#        test = test.drop(col,axis=1)
#        prediction = model.predict(test)
#        pred.append(prediction[col][0])
    
#    true = df[col]
#    acc_score = accuracy_score(true,pred)
#    print("Accuracy "+col+" :",acc_score)

for i in range(len(test_df)):
    test = pd.DataFrame(test_df.loc[i])
    test = test.T
    test = test.drop(label_cols,axis=1)
    prediction = model.predict(test)
    pred.append(prediction.values[0])
    
pred = np.array(pred)
pred.shape = (len(pred),13)
pred_df = pd.DataFrame(pred,columns = label_cols)

pred_df.to_csv('pred.csv',index = False)


print("Finished")