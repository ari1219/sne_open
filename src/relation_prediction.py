# coding: utf-8

import sys
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score as F1
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
import random
import argparse
import datetime

def main(args):
    # data_path
    train_file = "data/"+args.dataset+"_train.txt"
    test_file = "data/"+args.dataset+"_test.txt"
    emb_file = "embedding/"+args.emb_file_name+".pickle"

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

    # load embeddng
    with open(emb_file, "br") as f:
        emb = pickle.load(f)

    # preprocessing for train edges
    positive_train_edges = [edge for edge in train_edges if edge[2]==1]
    negative_train_edges = [[edge[0], edge[1], 0] for edge in train_edges if edge[2]==-1]
    sample_size = min([len(positive_train_edges), len(negative_train_edges)])
    sampled_edges = random.sample(positive_train_edges, sample_size) + negative_train_edges
    random.shuffle(sampled_edges)

    x_train = np.array([np.concatenate((emb[edge[0]], emb[edge[1]])) for edge in sampled_edges])
    y_train = [edge[2] for edge in sampled_edges]
    x_valid = np.array([np.concatenate((emb[edge[0]], emb[edge[1]])) for edge in test_edges])
    y_valid = [1 if edge[2]==1 else 0 for edge in test_edges]

    # train logisitic regression
    clf = LR().fit(x_train, y_train)

    # calc each metric and output log
    if args.read_log:
        log_name = "embedding/"+args.emb_file_name+"_emb_log.txt"
        with open(log_name, "r") as f:
            log = f.read()
    else:
        log = "No log"
    auc = roc_auc_score(y_valid, clf.predict_proba(x_valid)[:,1])
    print("auc", auc)
    y_ = clf.predict(x_valid)
    f1 = F1(y_valid, y_)
    print("F1", f1)
    macro = F1(y_valid, y_, labels=[0, 1], average="macro")
    print("macro F1", macro)
    if args.write_log:
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d%H:%M:%S")
        log_name = "log/"+now_time+"_relation_log.txt"
        with open(log_name, "w") as f:
            f.write("score\n")
            f.write("AUC:" + str(auc)+"\n")
            f.write("f1:" + str(f1) + "\n")
            f.write("macro f1:" + str(macro) + "\n")
            f.write("emb info\n")
            f.write(log)
    return auc, f1, macro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--emb_file_name", default="emb")
    parser.add_argument("--read_log", default=True)
    parser.add_argument("--write_log", default=True)
    args = parser.parse_args()
    data = args.dataset
    main(args)
