import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.models import BayesianModel
import pickle
from pgmpy.inference import VariableElimination
import logging
columns = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture','Support Devices']
       

for column in columns:
    print(column)
    filename = "_".join([w for w in column.split(" ")])
    with open('bayes_'+filename.lower()+'_model.pkl','rb') as f:
      model = pickle.load(f)
    inference = VariableElimination(model)
    max_p = inference.query(variables = ["labels"],evidence={'Sex':1})
    print("Sex : 1",max_p)
    max_p = inference.query(variables = ["labels"],evidence={'Sex':0})
    print("Sex : 0",max_p)
    for age in range(0,18):
        max_p = inference.query(variables = ["labels"],evidence={'Age':age})
        print("Age : "+age,max_p)