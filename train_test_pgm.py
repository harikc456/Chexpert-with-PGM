from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import pandas as pd
import numpy as np
from networkx.drawing.nx_pylab import draw_networkx
from networkx.algorithms.cycles import find_cycle
import random
import pickle
from sklearn.metrics import accuracy_score

num_nodes = 128

known_name_nodes = ['Sex','Age']
known_leaf_nodes = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
                    'Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis',
                    'Pneumothorax','Pleural Other','Fracture','Support Devices']
                    
def check_dag(edge_list):
    G = nx.DiGraph(edge_list)
    try:
        find_cycle(G, orientation='original')
    except:
        return False
    return True
    
def get_num_nodes(edge_list):
    G = nx.DiGraph(edge_list)
    return len(G.nodes)

def create_graph(list_of_nodes):
    num_nodes = len(list_of_nodes)
    list_of_edges = []
    i = 3 * num_nodes
    while i > 0 :
        node_num_parent = random.randint(0,num_nodes-1)
        node_num_child = random.randint(0,num_nodes-1)
        if node_num_parent == node_num_child:
            continue
        list_of_edges.append((list_of_nodes[node_num_parent],list_of_nodes[node_num_child]))
        cycle = check_dag(list_of_edges)
        if cycle:
            list_of_edges = list_of_edges[:-1]
        if get_num_nodes(list_of_edges) == num_nodes:
            break
        i = i - 1
    return list_of_edges
    
prefix = 'int_'
    
node_names = []
for i in range(1,num_nodes+1):
    node_names.append(prefix+str(i))
    
graph_edges = create_graph(node_names)

df = pd.read_csv('conv_encoded_images_128.csv')

new_graph_edges = graph_edges.copy()
for leaf in known_leaf_nodes:
    graph_edges = new_graph_edges.copy()
    new_edge_count = random.randint(0,int(0.5 * len(graph_edges)))
    nodes_to_be_added = known_name_nodes + [leaf]
    new_edges = [(i,leaf) for i in known_name_nodes]
    graph_edges = graph_edges + new_edges
    for i in range(new_edge_count):
        index1 = random.randint(0,len(node_names) - 1)
        index2 = random.randint(0,len(nodes_to_be_added) - 1)
        new_edge = (node_names[index1], nodes_to_be_added[index2])
        graph_edges.append(new_edge)
        cycle = check_dag(graph_edges)
        if cycle:
            count += 1
            graph_edges = graph_edges[:-1]
    
    
    #Creating model
    print("Creating model for predicting "+leaf)
    model = BayesianModel(graph_edges)
    train_df = df[list(model.nodes())]
    print(train_df.shape)
    model.fit(train_df)
    with open('bayes_model_pgmpy_'+leaf+'.pkl','wb') as f:
        pickle.dump(model,f)
        
    # testing model
    print("Testing model for predicting "+leaf)
    valid_df = pd.read_csv('conv_encoded_valid_images_128.csv')
    valid_df = valid_df[list(model.nodes())]
    pred = []
    for i in range(len(valid_df)):
        test = pd.DataFrame(valid_df.loc[i])
        test = test.T
        test = test.drop(leaf,axis=1)
        prediction = model.predict(test)
        pred.append(prediction[leaf][0])
    true = valid_df[leaf]
    #print(np.array(true).shape,np.array(pred).shape)
    acc_score = accuracy_score(true,pred)
    print("\n\n")
    print("Accuracy "+leaf+" :",acc_score)
