# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score as F1
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
import random
import networkx as nx
import sys

def main(data, n):
    # data path
    train_file = "data/"+data+"_train.txt"
    test_file = "data/"+data+"_test.txt"
    emb_file = "embedding/emb.pickle"

    # load train edges
    train_edges = []
    with open(train_file, "r") as f:
        line = f.readline()
        while line:
            train_edges.append(list(map(int, line.split("\n")[0].split(","))))
            line = f.readline()

    # load test edges
    test_edges = []
    with open(test_file, "r") as f:
        line = f.readline()
        while line:
            test_edges.append(list(map(int, line.split("\n")[0].split(","))))
            line = f.readline()

    # load embedding
    with open(emb_file, "br") as f:
        emb = pickle.load(f)

    # make grpah 
    G = nx.DiGraph()
    G.add_edges_from([[edge[0], edge[1]] for edge in train_edges])
    G.add_edges_from([[edge[0], edge[1]] for edge in test_edges])

    # preprocess test edge
    test_negative_edges = [[edge[0], edge[1], 0] for edge in test_edges if edge[2]==-1]
    test_positive_edges = [[edge[0], edge[1], 1] for edge in test_edges if edge[2]==1]
    # sample unexisting edges
    neg_edges = []
    while len(neg_edges) < len(test_negative_edges):
        edge = [np.random.randint(0, n-1), np.random.randint(0, n-1)]
        if G.has_edge(edge[0], edge[1]):
            continue
        else:
            neg_edges.append([edge[0], edge[1], 2])
    test_edges = test_negative_edges + test_positive_edges + neg_edges

    # preprocess train edge
    train_negative_edges = [[edge[0], edge[1], 0] for edge in train_edges if edge[2]==-1]
    # sample train edge (same number of negative edge)
    train_positive_edges = random.sample([edge for edge in train_edges if edge[2]==1], len(train_negative_edges))
    # sample unexisting edges
    neg_edges = []
    while len(neg_edges) < len(train_negative_edges):
        edge = [np.random.randint(0, n-1), np.random.randint(0, n-1)]
        if G.has_edge(edge[0], edge[1]):
            continue
        else:
            neg_edges.append([edge[0], edge[1], 2])
    sampled_edges = train_negative_edges+train_positive_edges+neg_edges
    random.shuffle(sampled_edges)

    # make train and test 
    x_train = np.array([np.concatenate((emb[edge[0]], emb[edge[1]])) for edge in sampled_edges])
    y_train = [edge[2] for edge in sampled_edges]
    x_valid = np.array([np.concatenate((emb[edge[0]], emb[edge[1]])) for edge in test_edges])
    y_valid = [edge[2] for edge in test_edges]

    # train logisitic regression
    clf = LR().fit(x_train, y_train)
    
    # calc each metric
    y_ = clf.predict(x_valid)
    macro = F1(y_valid, y_, labels=[0, 1, 2], average="macro")
    print("macro f1", macro)
    micro = F1(y_valid, y_, labels=[0, 1, 2], average="micro")
    print("micro f1", micro)
    return macro, micro

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
