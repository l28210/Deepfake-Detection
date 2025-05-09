import torch
import argparse
import os
import json


from datetime import datetime
from tools.utils import set_seed
from tools.datasets import ForgeryDataset
from torch.utils.data import DataLoader
from federation.fedavg import Fedavg

parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=10,
                    help='Number of clients. (default: 10)')
parser.add_argument('--num_rounds', type=int, default=5,
                    help='Number of rounds. (default: 5)')
parser.add_argument('--num_epochs', type=int, default=4,
                    help='Number of epochs. (default: 4)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size in each training step. (default: 8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. (default: 1e-3)')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='Learning rate decay.')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--dataset', type=str, default='archive', choices=['archive'])
parser.add_argument('--seed', type=int, default=666)
config = vars(parser.parse_args())

# 固定随机种子
set_seed(config['seed'])

# 选择运行的GPU
device = torch.device('cuda')

# tag取最后一段路径，无则使用时间作为tag
config['tag'] = config['tag'].split('/')[-1]
if not config['tag'].strip():
    config['tag'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# 保存参数文件，若已存在，则使用已存在的参数
log_dir = os.path.join(config['log_dir'], config['tag'])
os.makedirs(log_dir, exist_ok=True)
config_file = os.path.join(log_dir, 'config.json')
if os.path.exists(config_file):
    # Overwrite with saved config
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    # Save config to a JSON file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

# 每个 client 自己的数据
ds_train = []
ds_valid = []
ds_test = []
data_loader_train = []
data_loader_valid = []
data_loader_test = []

if config['dataset'] == 'archive':
    for client_id in range(config['num_clients']):
        ds_train.append(ForgeryDataset.example_gen('/home/l/test_self/deepfake_detect/data/archive/train'))
        ds_valid.append(ForgeryDataset.example_gen('/home/l/test_self/deepfake_detect/data/archive/valid'))
        ds_test.append(ForgeryDataset.example_gen('/home/l/test_self/deepfake_detect/data/archive/test'))

# 划分数据集到每个client中
for i in range(config['num_clients']):
    data_loader_train.append(DataLoader(
        ds_train[i],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    ))
    data_loader_valid.append( DataLoader(
        ds_valid[i],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    ))
    data_loader_test.append(DataLoader(
        ds_test[i],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    ))

# fed方法
federation = Fedavg(config, device, data_loader_train, data_loader_valid, data_loader_test)