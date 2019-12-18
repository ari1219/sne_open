# -*- coding:utf-8 -*-

import tensorflow as tf
import math
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx

# model of proposed method
class Model:

    def __init__(self,
                 n, # number of nodes in graph (contain training and test)
                 edges, # list of edges (only training edge)
                 d=30, # dimention of source/target embedding (output is d*2 dimentional vector)
                 lr=0.015, # learning rate
                 use_balance=True, # for relation loss (default True)
                 undersampling=False, # sampling positive edges use for each epoch (default False)
                 sampling_rate=3, # sampling rate of positive edges ()
                 k=10, # number of negative sampling 
                 alpha=0.5, # alpha 
                 batch_size=1000, # batch size
                 lam=1e-5): # reguralization params
        self.n = n
        self.edges = edges
        self.undersampling = undersampling
        if self.undersampling:
            self.positive_edge = [edge for edge in self.edges if edge[2]==1]
            self.negative_edge = [edge for edge in self.edges if edge[2]==-1]
            self.sampling_num = min(len(self.positive_edge), len(self.negative_edge)*sampling_rate)
        self.use_balance = use_balance
        self.dim = d
        self.lr = lr
        self.k = k
        self.alpha = alpha
        self.batch_size = batch_size
        self.lam = lam
        self.build_valiables()
        self.build_placeholder()
        self.build_model()
        self.build_train()

    def build_placeholder(self):
        self.u = tf.placeholder(tf.int32, shape=[None])
        self.v = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.int32, shape=[None])
        self.u_vec = tf.nn.embedding_lookup(self.u_emb, self.u)
        self.v_vec = tf.nn.embedding_lookup(self.v_emb, self.v)
        self.i = tf.reshape(tf.cast(self.r, tf.float32), shape=[-1, 1])
        self.pos_rate = tf.placeholder(tf.float32)
        self.neg_rate = tf.placeholder(tf.float32)

    def build_valiables(self):
        _bound = 6/math.sqrt(self.dim)
        self.u_emb = tf.Variable(tf.random_uniform([self.n, self.dim], minval=-_bound, maxval=_bound), dtype=tf.float32)
        self.v_emb = tf.Variable(tf.random_uniform([self.n, self.dim], minval=-_bound, maxval=_bound), dtype=tf.float32)

    def build_model(self):
        self.build_loss_skip()
        self.build_loss_rel()
        self.build_loss_reg()
        self.loss = (1-self.alpha)*self.loss_skip + self.alpha*self.loss_rel+self.lam*self.loss_reg

    # build Loss_structure
    def build_loss_skip(self):
        _bound = 6/math.sqrt(self.dim*2)
        self.w_pos = tf.Variable(tf.random_uniform([self.dim, self.dim], minval=-_bound, maxval=_bound), dtype=tf.float32)
        self.w_neg = tf.Variable(tf.random_uniform([self.dim, self.dim], minval=-_bound, maxval=_bound), dtype=tf.float32)
        self.b_1 = tf.Variable(tf.zeros([self.dim]), dtype=tf.float32)
        self.f = tf.where(self.r==1, tf.matmul(self.u_vec, self.w_pos), tf.matmul(self.u_vec, self.w_neg))+ self.b_1
        self.b = tf.Variable(tf.zeros([self.n]), dtype=tf.float32)
        self.loss_skip = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                self.v_emb,
                self.b,
                tf.reshape(self.v, shape=[-1, 1]),
                self.f,
                self.k,
                self.n
            ))

    # build Loss_relation
    def build_loss_rel(self):
        self.uv_vec = tf.concat([self.u_vec, self.v_vec], axis=1)
        self.w_3 = tf.Variable(tf.truncated_normal([2*self.dim, 2*self.dim], stddev=0.1), dtype=tf.float32)
        self.b_3 = tf.Variable(tf.zeros([2*self.dim]), dtype=tf.float32)
        self.h_2 = tf.sigmoid(tf.matmul(self.uv_vec, self.w_3) + self.b_3)
        self.w_4 = tf.Variable(tf.truncated_normal([2*self.dim, 1], stddev=0.1), dtype=tf.float32)
        self.b_4 = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        self.g = tf.sigmoid(tf.matmul(self.h_2, self.w_4)+self.b_4)
        self.reg_g = tf.clip_by_value(self.g, 1e-5, 1-1e-5) # for remove "nan" value
        if self.use_balance:
            self.loss_rel = - tf.reduce_sum(self.pos_rate*self.i*tf.log(self.reg_g) + self.neg_rate*(1-self.i)*tf.log(1-self.reg_g))
        else:
            self.loss_rel = - tf.reduce_mean(self.i*tf.log(self.reg_g) + (1-self.i)*tf.log(1-self.reg_g))

    def build_loss_reg(self):
        self.loss_reg = \
        tf.nn.l2_loss(self.u_emb)\
        +tf.nn.l2_loss(self.v_emb)\
        +tf.nn.l2_loss(self.w_pos)\
        +tf.nn.l2_loss(self.w_neg)\
        +tf.nn.l2_loss(self.b_1)\
        +tf.nn.l2_loss(self.w_3)\
        +tf.nn.l2_loss(self.b_3)\
        +tf.nn.l2_loss(self.w_4)\
        +tf.nn.l2_loss(self.b_4)

    def build_train(self):
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def variables_initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    def train_one_epoch(self, sess, ret_loss=True):
        for u, v, r in self.sample_iterater():
            if self.use_balance:
                pos_rate = 1/r.count(1)
                neg_rate = 1/r.count(0)
                feed_dict={self.u:u, self.v:v, self.r:r, self.pos_rate:pos_rate, self.neg_rate:neg_rate}
            else:
                pos_rate = 1
                neg_rate = 1
                feed_dict={self.u:u, self.v:v, self.r:r, self.pos_rate:1., self.neg_rate:1.}
            feed_dict={self.u:u, self.v:v, self.r:r, self.pos_rate:pos_rate, self.neg_rate:neg_rate}
            sess.run([self.train], feed_dict=feed_dict)


    def sample_iterater(self):
        if self.undersampling:
            datas = random.sample(self.positive_edge, self.sampling_num) + self.negative_edge
        else:
            datas = self.edges
        random.shuffle(datas)
        for i in range(0, len(datas), self.batch_size):
            u = [edge[0] for edge in datas[i:i+self.batch_size]]
            v = [edge[1] for edge in datas[i:i+self.batch_size]]
            r = [1 if edge[2]==1 else 0 for edge in datas[i:i+self.batch_size]]
            yield u, v, r

    def get_embedding(self, sess):
        return sess.run(tf.concat([self.u_emb, self.v_emb], 1))
