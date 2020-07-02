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
    
def check_dag(edge_list):
    G = nx.DiGraph(edge_list)
    try:
        cycle = find_cycle(G, orientation='original')
    except:
        return False
    return cycle

def generate_edges(child,parents):
    edge_list = []
    for parent in parents:
        edge_list.append([parent,child])
    return edge_list

def get_cycle_edge(cycle,score,list_of_edges):
    cycle_edge_scores = []
    for parent,child,direction in cycle:
        cycle_edge_scores.append(score.local_score(child,[parent]))
    min_index = cycle_edge_scores.index(min(cycle_edge_scores))
    low_score_edge = cycle[min_index][:-1]
    return low_score_edge

def get_model_architecture(df):
    score = K2Score(df)
    list_of_edges = []
    for i in df.columns:
        edge_scores = []
        for j in df.columns:
            if i!=j:
                sco =score.local_score(i,[j])
                edge_scores.append((i,j,sco))
        edge_scores.sort(key = lambda x:x[2],reverse = True)
        parents = [edge_scores[0][1]]
        best_score = edge_scores[0][2]
        for v in range(1,10):
            parents.append(edge_scores[v][1])
            new_score = score.local_score(i,parents)
            if new_score > best_score:
                best_score = new_score
            else:
                parents = parents[:-1]
                break
        list_of_edges += generate_edges(i,parents)
        cycle = check_dag(list_of_edges)
        while cycle:
            low_score_edge = get_cycle_edge(cycle,score,list_of_edges)
            list_of_edges.remove(list(low_score_edge))
            cycle = check_dag(list_of_edges)
    return list_of_edges

#columns = ['Consolidation','Fracture','Pneumothorax','Lung Lesion','Enlarged Cardiomediastinum',
#            'Pneumonia','Pleural Other','No Finding','Cardiomegaly','Lung Opacity',
#            'Edema','Pleural Effusion','Atelectasis','Support Devices']
            
columns = ['Fracture','Pneumothorax','Lung Lesion','Enlarged Cardiomediastinum','Consolidation']
           
string_cols = ['Sex', 'Age', 'labels']

for column in columns:
    print(column)
    filename = "_".join([w.lower() for w in column.split(" ")])
    edges = pd.read_csv(filename+'_edges.csv')
    edges = edges.drop(['Unnamed: 0'],axis = 1)
    data = pd.read_csv(filename+'_features.csv')
    new_cols = []
    for col in data.columns:
        if col not in string_cols:
            a = int(col)
            new_cols.append("X"+col)
        else:
            new_cols.append(col)
    data.columns = new_cols
    edges = edges.values
    edges = [tuple(edge)for edge in edges]
    bayes_model = BayesianModel(edges)
    bayes_model.fit(data)
    inference = BayesianModelSampling(bayes_model)
    evidence = [State('labels', 0)]
    zero_df = inference.likelihood_weighted_sample(evidence=evidence, size=10000, return_type='dataframe')
    evidence = [State('labels', 1)]
    one_df = inference.likelihood_weighted_sample(evidence=evidence, size=10000, return_type='dataframe')
    sample_df = pd.concat([zero_df,one_df]).sample(frac = 1)
    #if not {16,17,18}.issubset(set(sample_df['Age'])):
    #   for age in [16,17,18]:
    #       evidence = [State('Age', age)]
    #       age_df = inference.likelihood_weighted_sample(evidence=evidence, size=2000, return_type='dataframe')
    #       sample_df = pd.concat([sample_df,age_df]).sample(frac = 1)
    f = 'weighted_'+filename+'.csv'
    sample_df.to_csv(f,index = False)
    sample_df = sample_df.drop('_weight',axis = 1)
    feat_cols = sample_df.columns[sample_df.columns!='labels']
    X_train,X_test,y_train,y_test = train_test_split(sample_df[feat_cols],sample_df['labels'],test_size = 0.3,random_state = 43)
    X_train[column] = y_train
    list_of_edges = get_model_architecture(X_train)
    model = BayesianModel(list_of_edges)
    model.fit(X_train)
    accuracy = get_accuracy(model,X_train[feat_cols],X_train[column])
    print("Training")
    print("Accuracy",accuracy)
    accuracy = get_accuracy(model,X_test,y_test)
    print("Testing")
    print("Accuracy",accuracy)

    with open('weighted_bayes_'+filename.lower()+'_model.pkl','wb') as f:
        pickle.dump(model,f,protocol=4)