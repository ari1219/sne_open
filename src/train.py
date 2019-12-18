#-*- coding:utf-8 -*-

# program for training proposed model

from sne import Model
import tensorflow as tf
import pickle
import sys
from tqdm import tqdm

# params

epoch = 3 # number of epoch
d = 30 # dimention of source/target embedding (output is d*2 dimentional vector)
lr = 0.015 # learning rate
batch_size = 1000 # batch_size
alpha = 0.5 # alpha (balanced parameter)
k = 10 # number of negative sampling for Loss_structure
lam = 1e-5 # regralization parameter

output_file_path = "embedding/emb.pickle"

data = "data/"+sys.argv[1]+"_train.txt"
n = int(sys.argv[2])

# prepare data
with open(data, "r") as f:
    line = f.readline()
    edges = []
    while line:
        line = line.split("\n")[0]
        line = list(map(int, line.split(",")))
        edges.append(line)
        line = f.readline()

sne = Model(n, edges, d=d, alpha=alpha, k=k, lr=lr, lam=lam, batch_size=batch_size)
with tf.Session() as sess:
    sne.variables_initialize(sess)
    for i in tqdm(range(epoch)):
        sne.train_one_epoch(sess)
    x = sne.get_embedding(sess)
    with open(output_file_path, "bw") as f:
        pickle.dump(x, f)
