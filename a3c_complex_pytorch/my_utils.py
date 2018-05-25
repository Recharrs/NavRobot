import numpy as np

from gym import spaces
from baselines.common.vec_env import VecEnv
from unityagents import UnityEnvironment

# movie
# ================
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

def make_anim(images, fps=15, true_image=True):
    duration = len(images) / fps
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip

class movie_maker(object):
    def __init__(self, log_interval=100, path=''):
        self.log_interval = log_interval
        self.path = path
        self.frame = []
        self.episode = 0
        
    def add_new_image(self, image, isntruction):
        if self.episode % self.log_interval == 0:     
            from PIL import Image
            from PIL import ImageFont
            from PIL import ImageDraw 
            
            _image = np.swapaxes(image, 0, 2)
            _image = np.swapaxes(_image, 0, 1)
            
            data = np.array(_image * 255, dtype=np.uint8)
            img = Image.fromarray(data, 'RGB')
            
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("./Asset/Aaargh.ttf", 16)
            draw.text((0, 0), str(self.episode) + ':' + isntruction, (255,255,255), font=font)
            
            data = np.array(img, dtype=np.uint8)
            self.frame.append(data)

    def export_ani(self, inst, reward, sucessful_times):
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        if self.episode % self.log_interval == 0:
            #with open('./logs_1.txt', 'a') as logs_file:
            #    logs_file.write(str(self.episode) + ':' + str(reward) + ',' + str(sucessful_times) + '\n')
            
            clip = make_anim(self.frame)
            clip.write_videofile("./Asset/video/ep_{:04d}.mp4".format(self.episode), fps=15)

        self.frame = []
        self.episode += 1

# instr
# ================

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

# env
# ================

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

