import torch
import argparse
import os
import json
import glob
import time
import numpy as np

from collections import defaultdict
from datetime import datetime
from tools.utils import set_seed
from tools.datasets import ForgeryDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.replknet import create_RepLKNet31B
from sklearn.metrics import precision_score, recall_score, f1_score
from tools.eval_model import eval_model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size in each training step. (default: 32)')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--dataset', type=str, default='archive', choices=['archive'])
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--CPUs', type=int, default=16,
                    help='Number of CPU used for train.(default: 16)')
parser.add_argument('--mod', type=str, default='central', choices=['fed', 'central'], 
                    help='fed or central')
parser.add_argument('--clients_num', type=int, default=10,
                    help='number of clients')


config = vars(parser.parse_args())
config['tag'] = config['tag'].split('/')[-1]
if config['mod'] == 'central':
    log_dir = os.path.join(config['log_dir'], config['tag'])
elif config['mod'] == 'fed':
    log_dir = os.path.join(config['log_dir'], config['tag'], 'server')
else:
    print('error mode, please choose fed or central')
    exit()

