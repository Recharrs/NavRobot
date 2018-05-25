import os, sys
from contextlib import contextmanager
import time
import numpy as np

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

class Agent(object):
    def __init__(self, img_dim, inst_dim, n_action, name='model', sess=None):
        assert sess != None
        self.name = name
        self.sess = sess
        
        self.img_dim = img_dim
        self.inst_dim = inst_dim
        self.n_action = n_action
        
        self.X = tf.placeholder(tf.float32, [None] + img_dim, name='Observaion')
        self.I = tf.placeholder(tf.int32, [None, inst_dim], name='Instruction')
        self.y = tf.placeholder(tf.int32, [None], name='Expert')
        
        self._build_net(True, False)
        self._build_net(False, True)
        self._define_train_ops()

        tl.layers.initialize_global_variables(self.sess)

    def _build_net(self, is_train=True, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)

            n = InputLayer(self.X, name='in')
            n = Conv2d(n, 32, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c1/1')
            n = Conv2d(n, 32, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c1/2')
            n = MaxPool2d(n, (2, 2), (2, 2), 'VALID', name='max1')

            n = DropoutLayer(n, 0.75, is_fix=True, is_train=is_train, name='drop1')

            n = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c2/1')
            n = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, "VALID", name='c2/2')
            n = MaxPool2d(n, (2, 2), (2, 2), 'VALID', name='max2')
            # print(n.outputs)
            n = DropoutLayer(n, 0.75, is_fix=True, is_train=is_train, name='drop2')

            n = FlattenLayer(n, name='f')
            n = DenseLayer(n, 512, tf.nn.relu, name='dense1')
            n = DropoutLayer(n, 0.5, is_fix=True, is_train=is_train, name='drop3')
            n = DenseLayer(n, 3, tf.nn.tanh, name='o')
            
        if is_train:
            self.n_train = n
            self.y_predict = tf.argmax(n.outputs, axis=1)
        else:
            self.n_test = n
            self.y_predict = tf.argmax(n.outputs, axis=1)

    def _define_train_ops(self):
        self.cost = tl.cost.cross_entropy(self.n_train.outputs, self.y, 'entropy_loss')
        self.train_params = tl.layers.get_variables_with_name(self.name, train_only=True, printable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
            self.cost, var_list=self.train_params)

    def train(self, X, I, y, n_epoch=100, batch_size=10, print_freq=20):
        for epoch in range(n_epoch):
            start_time = time.time()
            total_err, n_iter = 0, 1
            
            for batch in self.get_minibatch(X, I, y, batch_size):
                _X, _I, _y = batch[:,0], batch[:,1], batch[:,2]
                _X = np.array([arr for arr in _X])
                _I = np.array([arr for arr in _I])
                
                _, err = self.sess.run([self.train_op, self.cost], 
                    feed_dict={self.X: _X, self.I: _I, self.y: _y})
                total_err += err
                n_iter += 1
            
            if epoch % print_freq == 0:
                size = 1024
#                 if len(X) > size:
#                     self.eval(X[-size:,:,:,:], I[-size:,:], y[-size:])
#                 else:
#                     self.eval(X, I, y)
                print("Epoch [%d/%d] cost:%f took:%fs" % 
                    (epoch, n_epoch, total_err/n_iter, time.time()-start_time))

    def predict(self, image, inst):
        #image_out, inst_out = self.sess.run([self.image_output, self.inst_output], 
        #    {self.X: image, self.I: inst})
        #print(np.linalg.norm(image_out), np.linalg.norm(inst_out))
        pi = self.sess.run(self.n_test.outputs, {self.X: image, self.I: inst})
        action = np.argmax(pi)
        return action
    
    def eval(self, ob, I, y):
        y_test, y_predict = self.sess.run([self.y, self.y_predict], 
                     feed_dict={self.X: ob, self.I: I, self.y: y})
        c_mat, f1, acc, f1_macro = tl.utils.evaluation(y_test, y_predict, 3)

    def get_minibatch(self, X, I, y, batch_size):
        dataset_0 = []
        dataset_1 = []
        dataset_2 = []
        
        dataset = np.array(list(zip(X, I, y)))
        np.random.shuffle(dataset)
        
        for _X, _I, _y in dataset:
            if _y == 0:   dataset_0.append([_X, _I, _y])
            elif _y == 1: dataset_1.append([_X, _I, _y])
            else:         dataset_2.append([_X, _I, _y])
        
        #size = min(len(dataset_0), len(dataset_1), len(dataset_2))
        dataset = dataset_0 + dataset_1 + dataset_2
        dataset = np.array(dataset)
        
        np.random.shuffle(dataset)
        size = len(dataset)
        minibatch = np.array_split(dataset, size//batch_size+1)
        return minibatch
        
    def save_model(self):
        tl.files.save_npz(self.n_test.all_params, name=self.name+'.npz', sess=self.sess)

    def load_model(self):
        tl.files.load_and_assign_npz(sess=self.sess, name=self.name+'.npz', network=self.n_test)
