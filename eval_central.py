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
from model.replknet import *

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

device = torch.device('cuda')

if config['dataset'] == 'archive':
    test_folder=os.path.join(config['folder_data'],'archive/test')
    ds_test=ForgeryDataset.example_gen(test_folder)

data_loader_test=DataLoader(
    ds_test,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['CPUs'],
    pin_memory=True,
)

os.environ['LARGE_KERNEL_CONV_IMPL'] = config['LARGE_KERNEL_CONV_IMPL']
checkpoints = sorted(glob(os.path.join(log_dir, '*.pth')))
ckpt = checkpoints[-1]
model = RepLKNet().load_state_dict(torch.load(ckpt))

metrics_dict = eval_model(model, data_loader_test, device)
os.makedirs(os.path.join(log_dir, "eval_metrics"), exist_ok=True)
with open(os.path.join(log_dir, "eval_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

print('\nDone.')