import pickle
import numpy as np
import pandas as pd
from pgmpy.sampling import GibbsSampling
from pgmpy.models import MarkovModel, BayesianModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_accuracy(model,X_test,y_test):
    pred = model.predict(X_test)
    return accuracy_score(y_test,pred)

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

columns = ['Fracture','Pleural Other','No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
          'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax',
           'Pleural Effusion','Support Devices']
           
columns = ['Pneumonia','Pleural Other','Consolidation','Fracture',
            'Pneumothorax','Lung Lesion','Enlarged Cardiomediastinum']
           
string_cols = ['Sex', 'Age', 'labels']
           
for column in columns:
    print(column)
    filename = "_".join([w.lower() for w in column.split(" ")])
    #edges = pd.read_csv(filename+'_edges.csv')
    #edges = edges.drop(['Unnamed: 0'],axis = 1)
    #data = pd.read_csv(filename+'_features.csv')
    #new_cols = []
    #for col in data.columns:
    #    if col not in string_cols:
    #        a = int(col)
    #        new_cols.append("X"+col)
    #    else:
    #        new_cols.append(col)
    #data.columns = new_cols

    #edges = edges.values
    #edges = [tuple(edge)for edge in edges]
    #bayes_model = BayesianModel(edges)
    #print("Starting Fitting the Data")
    #try:
    #    bayes_model.fit(data)
    #    gibbs = GibbsSampling(bayes_model)
    #except Exception as e:
    with open('bayes_'+filename.lower()+'_model.pkl','rb') as f:
        bayes_model = pickle.load(f)
    gibbs = GibbsSampling(bayes_model.to_markov_model())
    print("Starting the Sampling")
    sample_df = gibbs.sample(size=40000, return_type='dataframe')
    sample_df.to_csv("gibbs_"+filename+"_data.csv",index = False)
    feat_cols = new_df.columns[new_df.columns!=column]
    X_train,X_test,y_train,y_test = train_test_split(sample_df[feat_cols],sample_df[column],test_size = 0.3,random_state = 43)
    X_train[column] = y_train
    list_of_edges = get_model_architecture(feature_df)
    model = BayesianModel(list_of_edges)
    model.fit(X_train)
    #mm = model.to_markov_model()
    accuracy = get_accuracy(model,X_train[feature_cols],X_train['labels'])
    print("Training")
    print("Accuracy",accuracy)
    accuracy = get_accuracy(model,X_test,y_test)
    print("Testing")
    print("Accuracy",accuracy)

    with open('gibbs_bayes_'+filename.lower()+'_model.pkl','wb') as f:
        pickle.dump(model,f,protocol=4)