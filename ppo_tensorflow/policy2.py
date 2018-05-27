import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, conv_to_fc, fc, batch_to_seq, seq_to_batch, lstm
#from baselines.common.tf_util import conv2d
from baselines.common.distributions import make_pdtype

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# def fc(x, nh, name):
#     fan_in = nin = x.get_shape()[1].value
#     fan_out = nh
#     w_bound = np.sqrt(6. / (fan_in + fan_out))
#     w = tf.get_variable(name + "/w", [nin, nh], 
#         initializer=tf.random_uniform_initializer(-w_bound, w_bound))
#     b = tf.get_variable(name + "/b", [nh], initializer=tf.zeros_initializer())
#     return tf.matmul(x, w)+b

def nature_cnn(X):
    #x1 = tf.nn.relu(conv2d(X, 128, 'x1', filter_size=(8,8), stride=(4,4)))
    #x2 = tf.nn.relu(conv2d(x1, 64, 'x2', filter_size=(4,4), stride=(2,2)))
    #x3 = tf.nn.relu(conv2d(x2, 64, 'x3', filter_size=(4,4), stride=(2,2)))
    x1 = tf.nn.relu(conv( X, 'x1', nf=128, rf=8, stride=4, init_scale=1.0))
    x2 = tf.nn.relu(conv(x1, 'x2',  nf=64, rf=4, stride=2, init_scale=1.0))
    x3 = tf.nn.relu(conv(x2, 'x3',  nf=64, rf=4, stride=2, init_scale=1.0))
    return x3

class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False): 
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n

        X = tf.placeholder(tf.float32, ob_shape) #obs
        I = tf.placeholder(tf.int32, [nbatch, 5])
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        
        # Model
        with tf.variable_scope("model", reuse=reuse):
            # Image Processing
            with tf.variable_scope("cnn"):
                x_image_rep = nature_cnn(X)
            
            # Instructioin Processing
            with tf.variable_scope("GRU"):
                embedding = tf.get_variable(
                    'word_embedding', 
                    shape=[12, 32], 
                    initializer=tf.random_uniform_initializer(-1e-3, 1e-3)
                )
                gru_cell = tf.contrib.rnn.GRUCell(
                    num_units=256,
                    kernel_initializer=tf.random_uniform_initializer(-1e-3, 1e-3),
                    bias_initializer=tf.random_uniform_initializer(-1e-3, 1e-3)
                )

                encoder_hidden = gru_cell.zero_state(nbatch, dtype=tf.float32)
                for i in range(5):
                    word_embedding = tf.nn.embedding_lookup(embedding, I[:, i])
                    output, encoder_hidden = gru_cell.call(word_embedding, encoder_hidden)
                x_insts_rep = encoder_hidden

            # Gated-Attention layers
            with tf.variable_scope("x-attn"):
                x_attention = tf.sigmoid(fc(x_insts_rep, 'x-attn', 64, init_scale=1.0))
                x_attention = tf.expand_dims(x_attention, 1)
                x_attention = tf.expand_dims(x_attention, 2)
            
            with tf.variable_scope("Gated-Attention"):
                x = x_image_rep * x_attention
                x = conv_to_fc(x)
                x = tf.nn.relu(fc(x, 'x-Ga', 256, init_scale=1.0))
            
            xs = batch_to_seq(x, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h20, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm, init_scale=1.0)
            h20 = seq_to_batch(h20)
            
            with tf.variable_scope("pi"):
                pi = tf.layers.dense(h20, nact, 
                        kernel_initializer=normalized_columns_initializer(0.01))
            with tf.variable_scope("vf"):
                vf = tf.layers.dense(h20, 1, 
                        kernel_initializer=normalized_columns_initializer(0.01))

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, insts, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, I:insts, S:state, M:mask})

        def value(ob, insts, state, mask):
            return sess.run(v0, {X:ob, I:insts, S:state, M:mask})

        self.X = X
        self.I = I #
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        
        # start logging
        # =============
        if reuse:
            self.var_summary('./Asset/logdir', sess)
        
    def var_summary(self, path, sess):
        trainable_var = tf.trainable_variables(scope='model')
        
        self.cnn_var = [var for var in trainable_var if 'cnn' in var.name]
        self.gru_var = [var for var in trainable_var if 'GRU' in var.name]
        self.ga_var = [var for var in trainable_var if 'Gated-Attention' in var.name]
        self.lstm_var = [var for var in trainable_var if 'lstm1' in var.name]
        self.pi_var = [var for var in trainable_var if 'pi' in var.name]
        self.vf_var = [var for var in trainable_var if 'vf' in var.name]
        
        cnn_norm = tf.global_norm(self.cnn_var, name='cnn')
        gru_norm = tf.global_norm(self.gru_var, name='gru')
        ga_norm = tf.global_norm(self.ga_var, name='ga')
        lstm_norm = tf.global_norm(self.lstm_var, name='lstm')
        pi_norm = tf.global_norm(self.pi_var, name='pi')
        vf_norm = tf.global_norm(self.vf_var, name='vf')
        
        # add summary
        tf.summary.scalar('VarNorm/cnn', cnn_norm)
        tf.summary.scalar('VarNorm/gru', gru_norm)
        tf.summary.scalar('VarNorm/GA', ga_norm)
        tf.summary.scalar('VarNorm/lstm', lstm_norm)
        tf.summary.scalar('VarNorm/pi', pi_norm)
        tf.summary.scalar('VarNorm/vf', vf_norm)
