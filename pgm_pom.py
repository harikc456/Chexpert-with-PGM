from pomegranate import BayesianNetwork
import seaborn, time
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pomegranate import ConditionalProbabilityTable, DiscreteDistribution, State
seaborn.set_style('whitegrid')
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv('encoded_images.csv')

X_train = df[:int(len(df)*0.75)]
X_test = df[int(len(df)*0.75):]

print("Training")
model = BayesianNetwork.from_samples(X_train, algorithm='greedy')
print("Training Completed")
#print(model.structure)

with open('bayes_model.pkl','wb') as f:
  pickle.dump(model,f)
  
