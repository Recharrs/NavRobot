import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.tf_util import conv2d
from baselines.common.distributions import make_pdtype

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def fc(x, nh, name):
    fan_in = nin = x.get_shape()[1].value
    fan_out = nh
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    w = tf.get_variable(name + "/w", [nin, nh], 
        initializer=tf.random_uniform_initializer(-w_bound, w_bound))
    b = tf.get_variable(name + "/b", [nh], initializer=tf.zeros_initializer())
    return tf.matmul(x, w)+b

def nature_cnn(X):
    x1 = tf.nn.relu(conv2d(X, 128, 'x1', filter_size=(8,8), stride=(4,4)))
    x2 = tf.nn.relu(conv2d(x1, 64, 'x2', filter_size=(4,4), stride=(2,2)))
    x3 = tf.nn.relu(conv2d(x2, 64, 'x3', filter_size=(4,4), stride=(2,2)))
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
                    initializer=tf.random_uniform_initializer(-1, 1)
                )
                gru_cell = tf.contrib.rnn.GRUCell(num_units=256)

                encoder_hidden = gru_cell.zero_state(nbatch, dtype=tf.float32)
                for i in range(5):
                    word_embedding = tf.nn.embedding_lookup(embedding, I[:, i])
                    output, encoder_hidden = gru_cell.call(word_embedding, encoder_hidden)
                x_insts_rep = encoder_hidden

            # Gated-Attention layers
            with tf.variable_scope("Gated-Attention"):
                x_attention = tf.sigmoid(fc(x_insts_rep, 64, 'x-attention'))
                x_attention = tf.expand_dims(x_attention, 1)
                x_attention = tf.expand_dims(x_attention, 2)

                x = x_image_rep * x_attention                
                x = conv_to_fc(x)
                x = tf.nn.relu(fc(x, 256, 'x-Gated-Attention'))
            
            xs = batch_to_seq(x, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h20, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h20 = seq_to_batch(h20)
            
            pi = tf.layers.dense(h20, nact, 
                    kernel_initializer=normalized_columns_initializer(0.01))
            vf = tf.layers.dense(h20, 1, 
                    kernel_initializer=normalized_columns_initializer(1.0))

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