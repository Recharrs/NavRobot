import numpy as np
import torch
import torch.nn.functional as F

from models import *
from models import A3C_LSTM_GA

class test:
    def __init__(self, args, share_model):
        torch.manual_seed(args.seed + 0)

        self.model = A3C_LSTM_GA(args)
        self.model = self.model.cuda()
        self.word_to_idx = {
            'object': 4, 'cylinder': 5, 'blue': 7, 'Go': 0, 'cube': 8, 
            'green': 9, 'ball': 6, 'red': 3, 'the': 2, 'yellow': 10, 'to': 1,
            'go': 11, 'any': 12, 'then': 13, 'sphere': 6
        }

        if (args.load != "0"):
            print("Loading model ... "+ args.load)
            self.model.load_state_dict(
                torch.load(args.load, map_location=lambda storage, loc: storage))
        self.model.eval()
        
        self.start = True
        self.episode_length = 0
    
    def step(self, image, instruction):
        # Getting indices of the words in the instruction
        instruction_idx = []
        for word in instruction.split(" "):
            instruction_idx.append(self.word_to_idx[word])
        instruction_idx = np.array(instruction_idx)

        image = torch.from_numpy(image).float()
        instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
        
        if self.start:
            self.cx = Variable(torch.zeros(1, 256), volatile=True).cuda()
            self.hx = Variable(torch.zeros(1, 256), volatile=True).cuda()
            self.start = False
        
        self.tx = Variable(torch.from_numpy(np.array([self.episode_length])).long(),
                      volatile=True)
        self.episode_length += 0
        
        value, logit, (self.hx, self.cx) = self.model(
                (Variable(image.unsqueeze(0), volatile=True).cuda(),
                 Variable(instruction_idx, volatile=True).cuda(),
                (self.tx, self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].cpu().data.numpy()

        return action
