# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score as F1
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
import random
import networkx as nx
import sys
import argparse
import datetime

def main(args):
    # data path
    train_file = "data/"+args.dataset+"_train.txt"
    test_file = "data/"+args.dataset+"_test.txt"
    emb_file = "embedding/"+args.emb_file_name+".pickle"

    nodes = set()
    # load train edges
    train_edges = []
    with open(train_file, "r") as f:
        line = f.readline()
        while line:
            line = list(map(int, line.split("\n")[0].split(",")))
            train_edges.append(line)
            nodes.add(line[0])
            nodes.add(line[1])
            line = f.readline()

    # load test edges
    test_edges = []
    with open(test_file, "r") as f:
        line = f.readline()
        while line:
            line = list(map(int, line.split("\n")[0].split(",")))
            test_edges.append(line)
            nodes.add(line[0])
            nodes.add(line[1])
            line = f.readline()

    n = len(nodes)

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
    clf = LR(multi_class="ovr").fit(x_train, y_train)

    # calc each metric and output log
    y_ = clf.predict(x_valid)
    if args.read_log:
        log_name = "embedding/"+args.emb_file_name+"_emb_log.txt"
        with open(log_name, "r") as f:
            log = f.read()
    else:
        log = "No log"
    macro = F1(y_valid, y_, labels=[0, 1, 2], average="macro")
    print("macro f1", macro)
    micro = F1(y_valid, y_, labels=[0, 1, 2], average="micro")
    print("micro f1", micro)
    if args.write_log:
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d%H:%M:%S")
        log_name = "log/"+now_time+"_link_prediction_log.txt"
        with open(log_name, "w") as f:
            f.write("score\n")
            f.write("micro f1:" + str(micro) + "\n")
            f.write("macro f1:" + str(macro) + "\n")
            f.write("emb info\n")
            f.write(log)
    return macro, micro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--emb_file_name", default="emb")
    parser.add_argument("--read_log", default=True)
    parser.add_argument("--write_log", default=True)
    args = parser.parse_args()
    data = args.dataset
    main(args)
