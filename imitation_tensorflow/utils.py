import numpy as np

from gym import spaces
from baselines.common.vec_env import VecEnv
from unityagents import UnityEnvironment

# Make animation
# ===============
def make_anim(images, fps=60, true_image=False):
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
            return (x * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip

def plt_vedio(images, instruction, reward):
    # export
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.tight_layout()
    ims = []

    color = 'orange'
    if reward == 1.0:
        color = 'green'
    elif reward == -0.2: 
        color = 'red'

    import matplotlib.patches as mpatches        
    import matplotlib.animation as animation
    for frame in images:
        patch = mpatches.Patch(color=color, label=instruction)
        plt.legend(handles=[patch])
        plt.axis('off')
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    
    plt.close(fig)
    ani = animation.ArtistAnimation(fig, ims, interval=90, blit=True)
    #ani.save('dynamic_images.mp4')
    return ani

class movie_maker:
    def __init__(self, log_interval=10, path=''):
        self.log_interval = log_interval
        self.path = path
        self.frame = []
        self.episode = 0
        
    def add_new_image(self, image):
        self.frame.append(image)
        
    def export_ani(self, image, inst, reward):
        if self.episode % self.log_interval == 0:        
            export_path = self.path + '/ep_{:04d}.mp4'.format(self.episode)
            ani = plt_vedio(self.frame, idx2word(inst), reward)
            ani.save(export_path)
        self.frame = []
        self.episode += 1

# IDX convertion
# ===============
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

def word2idx(inst):
    word_to_idx = {
          'object': 4, 'cylinder': 5, 'blue': 7, 'Go': 0, 'cube': 8, 
          'green': 9, 'ball': 6, 'red': 3, 'the': 2, 'yellow': 10, 'to': 1,
          'go': 11, 'any': 12, 'then': 13, 'sphere': 6
      }

    instruction_idx = []
    for word in inst.split(" "):
        instruction_idx.append(word_to_idx[word])

    return instruction_idx

def idx2word(idx):
    word_to_idx = {
          'object': 4, 'cylinder': 5, 'blue': 7, 'Go': 0, 'cube': 8, 
          'green': 9, 'ball': 6, 'red': 3, 'the': 2, 'yellow': 10, 'to': 1,
          'go': 11, 'any': 12, 'then': 13, 'sphere': 6
    }
    inv_map = {v: k for k, v in word_to_idx.items()}
    
    instruction_idx = []
    for i in idx:
        instruction_idx.append(inv_map[i])

    return " ".join(instruction_idx)

# For ENV
# ===============
class env_wrapper(object):
    def __init__(self, app_name, idx=0, train_mode=True):
        # Unity scene
        self.env = UnityEnvironment(file_name=app_name, worker_id=idx)
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
    
    def step(self, act):
        env_info = self.env.step(act)[self.default_brain]
        ob     = env_info.observations[0][0]
        inst   = env_info.states[0]
        reward = env_info.rewards[0]
        done   = env_info.local_done[0]
        return ob, inst, reward, done, dict()
        
    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
        obs   = env_info.observations[0]
        insts = env_info.states
        return obs[0], insts[0]
    
    def close(self):
        self.env.close()

class multi_env(VecEnv):
    def __init__(self, app_name, num_envs=2, base=0):
        self.name = app_name
        self.envs = [env_wrapper(app_name, base+idx) for idx in range(num_envs)]
        self.num_envs = num_envs
        
        env = self.envs[0]
        self.observation_space = env.ob_space
        self.action_space = env.ac_space
        VecEnv.__init__(self, num_envs, env.ob_space, env.ac_space)
        
        self.ts = np.zeros(num_envs, dtype='int')  
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
    
    def step_wait(self):
        results = [env.step(action) for (action, env) in zip(self.actions, self.envs)]
        obs, insts, rews, dones, infos = map(np.array, zip(*results))

        # maximum steps
        self.ts += 1
        for  (i, t) in enumerate(self.ts):
            if t > 150:
                dones[i] = True
                self.ts[i] = 0
        
        # change reward
        for (i, done) in enumerate(dones):
            if done:
                (obs[i], insts[i]) = self.envs[i].reset()
                self.ts[i] = 0
        
        self.actions = None
        return (np.array(obs), np.array(insts)), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, insts = map(np.array, zip(*results))
        return obs, insts

    def close(self):
        for env in self.envs: env.close()
        