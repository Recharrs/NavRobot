import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        
        self.td_map = None
        def train(lr, cliprange, obs, insts, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, train_model.I:insts,
                    A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            self.td_map = td_map
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        
        self.loss_names = [
            'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101
        
        # add summary
        # ===========
        self.writer = tf.summary.FileWriter('./Asset/logdir', sess.graph)
        
        cnn_grads = tf.gradients(loss, train_model.cnn_var)
        gru_grads = tf.gradients(loss, train_model.gru_var)
        ga_grads = tf.gradients(loss, train_model.ga_var)
        lstm_grads = tf.gradients(loss, train_model.lstm_var)
        pi_grads = tf.gradients(loss, train_model.pi_var)
        vf_grads = tf.gradients(loss, train_model.vf_var)

        cnn_grad_norm = tf.global_norm(cnn_grads, name='cnn_grads')
        gru_grad_norm = tf.global_norm(gru_grads, name='gru_grads')
        ga_grad_norm = tf.global_norm(ga_grads, name='ga_grads')
        lstm_grad_norm = tf.global_norm(lstm_grads, name='lstm_grads')
        pi_grad_norm = tf.global_norm(pi_grads, name='pi_grads')
        vf_grad_norm = tf.global_norm(vf_grads, name='vf_grads')
        
        tf.summary.scalar('GradNorm/cnn', cnn_grad_norm)
        tf.summary.scalar('GradNorm/gru', gru_grad_norm)
        tf.summary.scalar('GradNorm/GA', ga_grad_norm)
        tf.summary.scalar('GradNorm/lstm', lstm_grad_norm)
        tf.summary.scalar('GradNorm/pi', pi_grad_norm)
        tf.summary.scalar('GradNorm/vf', vf_grad_norm)
        
        tf.summary.scalar('loss/policy_loss', pg_loss)
        tf.summary.scalar('loss/value_loss', vf_loss)
        tf.summary.scalar('loss/entropy', entropy)
        
        self.merged = tf.summary.merge_all()
        
        def get_summary():          
            return sess.run(self.merged, self.td_map)
        self.get_summary = get_summary
            
        