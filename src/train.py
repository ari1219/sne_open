#-*- coding:utf-8 -*-

from sne import Model
import tensorflow as tf
import pickle
from tqdm import tqdm
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("dataset")
parser.add_argument("--epoch", default=3)
parser.add_argument("--d", default=60)
parser.add_argument("--lr", default=0.015)
parser.add_argument("--batch_size", default=1000)
parser.add_argument("--alpha", default=0.5)
parser.add_argument("--k", default=10)
parser.add_argument("--lam", default=1e-5)
parser.add_argument("--emb_file_name", default="emb")
parser.add_argument("--output_log", default=True)

args = parser.parse_args()

# parameter setting
epoch = int(args.epoch) # number of epoch
d = int(args.d) # dimention of source/target embedding
lr = float(args.lr) # learning rate
batch_size = int(args.batch_size) # batch_size
alpha = float(args.alpha) # alpha (balanced parameter)
k = int(args.k) # number of negative sampling for Loss_structure
lam = float(args.lam) # regralization parameter
output_file_path = "embedding/"+args.emb_file_name+".pickle"

# dataset
train_data = "data/"+args.dataset+"_train.txt"
test_data = "data/"+args.dataset+"_test.txt"

# prepare data
nodes = set()
with open(train_data, "r") as f:
    line = f.readline()
    edges = []
    while line:
        line = line.split("\n")[0]
        line = list(map(int, line.split(",")))
        edges.append(line)
        nodes.add(line[0])
        nodes.add(line[1])
        line = f.readline()
with open(test_data, "r") as f:
    line = f.readline()
    while line:
        line = line.split("\n")[0]
        line = list(map(int, line.split(",")))
        nodes.add(line[0])
        nodes.add(line[1])
        line = f.readline()
n = len(nodes)

# call and train models
sne = Model(n, edges, d=int(d/2), alpha=alpha, k=k, lr=lr, lam=lam, batch_size=batch_size)
with tf.Session() as sess:
    # initialize learning values
    sne.variables_initialize(sess)
    start = time.time()
    # training
    for i in range(epoch):
        sne.train_one_epoch(sess)
    end = time.time()
    # output embedding
    x = sne.get_embedding(sess)
    with open(output_file_path, "bw") as f:
        pickle.dump(x, f)
    # output log file
    if args.output_log:
        log_name = "embedding/" + args.emb_file_name + "_emb_log.txt"
        with open(log_name, "w") as f:
            f.write("train dataset:data/"+args.dataset+"_train.txt\n")
            f.write("test dataset:data/"+args.dataset+"_test.txt\n")
            f.write("output_file_path:"+output_file_path+"\n")
            f.write("epoch:"+str(epoch)+"\n")
            f.write("d:"+str(d)+"\n")
            f.write("lr:"+str(lr)+"\n")
            f.write("batch_size:"+str(batch_size)+"\n")
            f.write("alpha:"+str(alpha)+"\n")
            f.write("k:"+str(k)+"\n")
            f.write("lam:"+str(lam)+"\n")
            f.write("learning time:"+str(end-start)+"\n")
