import numpy as np
import pandas as pd
import pickle
from pgmpy.sampling import GibbsSampling
from pgmpy.models import MarkovModel, BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from networkx.algorithms.cycles import find_cycle
from pgmpy.estimators import BdeuScore, K2Score, BicScore

def get_accuracy(model,X_test,y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test,pred)
    
columns = ['Consolidation','Fracture','Pneumothorax','Lung Lesion','Enlarged Cardiomediastinum',
            'Pneumonia','Pleural Other','No Finding','Cardiomegaly','Lung Opacity',
            'Edema','Pleural Effusion','Atelectasis','Support Devices']
            
           
string_cols = ['Sex', 'Age', 'labels']

for column in columns:
    print(column)
    filename = "_".join([w.lower() for w in column.split(" ")])
    sample_df = pd.read_csv('rejection_sampled_'+filename+'.csv')
    feat_cols = sample_df.columns[sample_df.columns!='labels']
    X_train,X_test,y_train,y_test = train_test_split(sample_df[feat_cols],sample_df['labels'],test_size = 0.3,random_state = 43)
    X_train[column] = y_train
    
    with open('rejection_bayes_'+filename.lower()+'_model.pkl','rb') as f:
        model = pickle.load(f)
    
    accuracy = get_accuracy(model,X_train[feat_cols],X_train[column])
    print("Training")
    print("Accuracy",accuracy)
    accuracy = get_accuracy(model,X_test,y_test)
    print("Testing")
    print("Accuracy",accuracy)

    