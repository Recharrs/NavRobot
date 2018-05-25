import os, sys
import numpy as np
import tensorflow as tf
import random

from model import Agent
import utils

app_path = '../Apps/kobuki_ir2/kobuki_ir2'
video_path = './Asset/video'

# train config
n_episode = int(10e4)
max_step = 150
beta = 0.5

# model config
img_dim = [64, 64, 3]
inst_dim = 5
n_action = 3
batch_size = 1024
n_epoch = 100

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Model
    # =====================
    sess = tf.InteractiveSession()
    env = utils.env_wrapper(app_path, idx=1024)
    model = Agent(img_dim, inst_dim, n_action, sess=sess)
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
    collect_num_envs = 16
    collect_data_env = utils.multi_env(app_path, num_envs=collect_num_envs)
    
    obs, states = collect_data_env.reset()
    insts = [utils.word2idx(utils.id2str(state[0:3])) for state in states]
    actions = [state[3] for state in states]    
    for _ in range(300):
        (obs, states), reward, done, _ = collect_data_env.step(actions)
        
        insts = [utils.word2idx(utils.id2str(state[0:3])) for state in states]
        actions = [state[3] for state in states]
        print(actions)

        images_all = np.concatenate([images_all, obs], axis=0)
        insts_all = np.concatenate([insts_all, insts], axis=0)
        actions_all = np.concatenate([actions_all, actions], axis=0)        
    
        # Count DATA
        # =====================
        actions_all_2 = actions_all.tolist()
        print("#data: ", len(actions_all_2))
        print("#action 0: ", actions_all_2.count(0))
        print("#action 1: ", actions_all_2.count(1))
        print("#action 2: ", actions_all_2.count(2))
    
    collect_data_env.close()
    
    # Pretrain model using data for demonstration
    # =====================
    #model.load_model()
    model.train(images_all, insts_all, actions_all, n_epoch=n_epoch, batch_size=batch_size)
    model.save_model()

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
        expert_action = 0
        for i in range(max_step):
            act_agent = model.predict([ob], [inst])
            act = act_agent if beta < random.uniform(0,1) else expert_action
            ob, state, reward, done, _ = env.step(act)
            inst_prev = inst
            inst = utils.word2idx(utils.id2str(state[0:3]))
            expert_action = state[3]
            
            if done or i == max_step - 1:
                mv_maker.export_ani(ob, inst_prev, reward)
                break
            else:
                mv_maker.add_new_image(ob)
                ob_list.append(ob)
                inst_list.append(inst)
                action_list.append(expert_action)
            reward_sum += reward
        
        beta = beta * 0.5
        
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
        
#         if len(images_all) > 1024:
#             images_all = images_all[-1024:,:,:,:]
#             insts_all = insts_all[-1024:,:]
#             actions_all = actions_all[-1024:]
        
        # Train Model
        # =====================
        model.train(images_all, insts_all, actions_all, n_epoch=n_epoch, batch_size=batch_size)
        model.save_model()
