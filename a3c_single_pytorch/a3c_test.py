import numpy as np
import torch
import torch.nn.functional as F
import time
import logging

from models import A3C_LSTM_GA

from torch.autograd import Variable
from constants import *

from Asset.my_utils import *

class test:
    def __init__(self, args, share_model):
        torch.manual_seed(args.seed + rank)

        self.model = A3C_LSTM_GA(args)
        self.model = model.cuda()

        if (args.load != "0"):
            print("Loading model ... "+ args.load)
            self.model.load_state_dict(
                torch.load(args.load, map_location=lambda storage, loc: storage))
        model.eval()
        
        self.start = True
    
    def step(image, instruction):
        # Getting indices of the words in the instruction
        instruction_idx = []
        for word in instruction.split(" "):
            instruction_idx.append(env.word_to_idx[word])
        instruction_idx = np.array(instruction_idx)

        image = torch.from_numpy(image).float()
        instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
        
        if self.start:
            self.cx = Variable(torch.zeros(1, 256), volatile=True)
            self.hx = Variable(torch.zeros(1, 256), volatile=True)
            self.start = False

        self.tx = Variable(torch.from_numpy(np.array([episode_length])).long(),
                      volatile=True)

        value, logit, (self.hx, self.cx) = model(
                (Variable(image.unsqueeze(0), volatile=True),
                 Variable(instruction_idx, volatile=True), (tx, hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        return action
