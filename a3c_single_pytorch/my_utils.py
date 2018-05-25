import numpy as np

from gym import spaces
from baselines.common.vec_env import VecEnv
from unityagents import UnityEnvironment

def make_plt_anim(images, instruction, reward):
	# export
	import matplotlib
	matplotlib.use('agg')
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ims = []
	
	color = 'orange'
	if reward == 1.0: 
			color = 'green'
	elif reward == -0.2:
			color = 'red'

	import matplotlib.patches as mpatches        
	import matplotlib.animation as animation
	for frame in images:
		red_patch = mpatches.Patch(color=color, label=instruction)
		plt.legend(handles=[red_patch])
		plt.axis('off')

		im = plt.imshow(frame, animated=True)
		ims.append([im])
	
	plt.close(fig)

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
	#ani.save('dynamic_images.mp4')
	return ani
    
from baselines.common.vec_env import VecEnv
from unityagents import UnityEnvironment

dict_action = {
	0: "none",
	1: "go to the"
}

dict_color = {
	0: "any",
	1: "red",
	2: "green",
	3: "blue",
	4: "yellow"
}

dict_obj = {
	0: "object",
	1: "cube",
	2: "cylinder",
	3: "sphere"
}

def id2str(idx):
    inst = [dict_action[idx[0]], dict_color[idx[1]], dict_obj[idx[2]]]
    return " ".join(inst)

# def id2str(idx):
#     print(idx)
#     import json
#     data_train = json.load(open('./Asset/instructions/instructions_train.json'))
#     data_train_all = data_train['allNLPData']
#     return data_train_all[int(idx)]['instruction']

class env_wrapper(object):
    def __init__(self, app_name, idx=0, base=0, train_mode=True):
        # Unity scene
        self.env = UnityEnvironment(file_name=app_name, worker_id=idx+base)
        self.train_mode = train_mode
        self.name = app_name

        # default brain
        self.default_brain = self.env.brain_names[0]
        brain = self.env.brains[self.default_brain]
        env_info = self.env.reset()[self.default_brain]

        # OpenAI-gym
        self.num_envs = 1
        self.action_space = spaces.Discrete(brain.action_space_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=env_info.observations[0][0].shape)
        self.ac_space = self.action_space
        self.ob_space = self.observation_space
    
        # instruction only
        self.word_to_idx = {
            'object': 4, 'cylinder': 5, 'blue': 7, 'Go': 0, 'cube': 8, 
            'green': 9, 'ball': 6, 'red': 3, 'the': 2, 'yellow': 10, 'to': 1,
            'go': 11, 'any': 12, 'then': 13, 'sphere': 6
        }

    def step(self, act):
        env_info = self.env.step(act)[self.default_brain]
        ob       = env_info.observations[0][0]
        inst     = env_info.states[0]
        reward   = env_info.rewards[0]
        done     = env_info.local_done[0]
        
        ob = np.swapaxes(ob, 0, 1)
        ob = np.swapaxes(ob, 0, 2)
        inst = id2str(inst)
        
        return (ob, inst), reward, done, dict()
        
    def reset(self):        
        env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
        ob     = env_info.observations[0][0]
        inst = env_info.states[0]
        
        ob = np.swapaxes(ob, 0, 1)
        ob = np.swapaxes(ob, 0, 2)
        inst = id2str(inst)
        
        return (ob, inst), None, None, None
    
    def close(self):
        self.env.close()

