import argparse
import os
import numpy as np
import torch
import torch.multiprocessing as mp

from models import A3C_LSTM_GA
from a3c_train import train
from a3c_test import test

import logging

from my_utils import *

parser = argparse.ArgumentParser(description='Gated-Attention for Grounding')

# Environment arguments
parser.add_argument('-l', '--max-episode-length', type=int, default=150,
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-d', '--difficulty', type=str, default="hard",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
parser.add_argument('--living-reward', type=float, default=-0.005,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300,
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=0,
                    help="""Visualize the envrionment (default: 0,
                    use 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('--scenario-path', type=str, default="maps/room.wad",
                    help="""Doom scenario file to load
                    (default: maps/room.wad)""")
parser.add_argument('--interactive', type=int, default=0,
                    help="""Interactive mode enables human to play
                    (default: 0)""")
parser.add_argument('--all-instr-file', type=str,
                    default="data/instructions_all.json",
                    help="""All instructions file
                    (default: data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json",
                    help="""Test instructions file
                    (default: data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str,
                    default="data/object_sizes.txt",
                    help='Object size file (default: data/object_sizes.txt)')

# A3C arguments
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')

parser.add_argument('-e', '--evaluate', type=int, default=0,
                    help="""0:Train, 1:Evaluate MultiTask Generalization
                    2:Evaluate Zero-shot Generalization (default: 0)""")
                    
parser.add_argument('--dump-location', type=str, default="./Asset/model/saved_2/model",
                    help='path to dump models and log (default: ./saved/)')

# Self-defined arguments
parser.add_argument('--load', type=str, default="./Asset/model/saved_model",
                    help='model path to load, 0 to not reload (default: 0)')
parser.add_argument('-n', '--num-processes', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--app-location', type=str, default='../Apps/kobuki_c2/kobuki_c2',
                    help='location of appilication')
parser.add_argument('--model-location', type=str, default='./Asset/model/saved_model',
                    help='location of model')
parser.add_argument('--video-location', type=str, default='./Asset/video',
                    help='location of appilication')

word_to_idx = {
    'object': 4, 'cylinder': 5, 'blue': 7, 'Go': 0, 'cube': 8, 
    'green': 9, 'ball': 6, 'red': 3, 'the': 2, 'yellow': 10, 'to': 1,
    'go': 11, 'any': 12, 'then': 13, 'sphere': 6
}
    

class Model:    
    def __init__(self):
        args = parser.parse_args()
        args.input_size = len(word_to_idx)

        shared_model = A3C_LSTM_GA(args)
        shared_model = shared_model.cuda()

        # Load the model
        if (args.load != "0"):
            shared_model.load_state_dict(
                torch.load(args.load, map_location=lambda storage, loc: storage))

        shared_model.share_memory()
        self.test_model = test(args, shared_model)
    
    def step(self, image, instruction):
        action = self.test_model.step(image, instruction)
        return action

if __name__ == '__main__':
    test_model = Model()
    
    while True:
        image=np.zeros((3, 168, 300))
        instruction='go to the red ball'
        action = test_model.step(image, instruction)
        print(action)
