import os, sys
import numpy as np
import tensorflow as tf

from model import Agent
import utils

app_path = '../Apps/kobuki_d2/kobuki_d2'
video_path = './Asset/video'

# train config
n_episode = int(10e6)
steps = 300

# model config
img_dim = [168, 300, 3]
inst_dim = 5
n_action = 3
batch_size = 32
n_epoch = 30

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Model
    # =====================
    sess = tf.InteractiveSession()
    env = utils.env_wrapper(app_path, idx=1024)
    model = Agent(img_dim, inst_dim, n_action, nbatch=batch_size, sess=sess)
    output_file = open('results.txt', 'w')
    mv_maker = utils.movie_maker(log_interval=1, path=video_path)
    
    # DataSet
    # =====================
    images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
    insts_all  = np.zeros((0, inst_dim))
    actions_all = np.zeros((0))
    rewards_all = np.zeros((0,))
    
    # Get demonstration data
    # =====================
    print("#"*50)
    print('Collecting data from teacher (fake AI) ... ')
    
    action = 0
    for i in range(steps):
        if i == 0:
            ob, state = env.reset()
            inst = utils.word2idx(utils.id2str(state[0:3]))
            action = state[3]
        else:
            ob, state, reward, done, _ = env.step(action)
            inst = utils.word2idx(utils.id2str(state[0:3]))
            action = state[3]        

            images_all = np.concatenate([images_all, [ob]], axis=0)
            insts_all = np.concatenate([insts_all, [inst]], axis=0)
            actions_all = np.concatenate([actions_all, [action]], axis=0)        

    # Pretrain model using data for demonstration
    # =====================
    model.load_model()

    # Aggregate and retrain
    # =====================    
    for episode in range(n_episode):
        ob_list = []
        inst_list = []
        action_list = []
        reward_list = []
        
        # restart the game for every episode
        (ob, state) = env.reset()
        inst = utils.word2idx(utils.id2str(state[0:3]))
        expert_action = state[3]
        reward_sum = 0.0
        
        print(" "*50)
        print("#"*50)
        print("# Episode: %d start" % episode)
        
        # do an episode
        for i in range(steps):
            act = model.predict([ob], [inst])
            ob, state, reward, done, _ = env.step(act)
            inst_prev = inst
            inst = utils.word2idx(utils.id2str(state[0:3]))
            expert_action = state[3]
            
            if done or i == steps - 1:
                mv_maker.export_ani(ob, inst_prev, reward)
                break
            else:
                mv_maker.add_new_image(ob)
                ob_list.append(ob)
                inst_list.append(inst)
                action_list.append(expert_action)
            reward_sum += reward
        
        # print result
        print("# step: %d reward: %f " % (i, reward_sum))
        print("#"*50)
        output_file.write('Number of Steps: %02d\t Reward: %0.04f\n' % (i, reward_sum))
        
        # Dataset AGGregation: bring learner’s and expert’s trajectory distributions
        # closer by labelling additional data points resulting from applying the current policy
        # =====================
        images_all = np.concatenate([images_all, ob_list], axis=0)
        insts_all = np.concatenate([insts_all, inst_list], axis=0)
        actions_all = np.concatenate([actions_all, action_list], axis=0)

