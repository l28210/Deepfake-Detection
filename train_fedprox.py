from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from tools.datasets.wads_fed import WADS_FED
from tools.datasets.wads_only import WADS_ONLY
from tools.datasets.snowykitti_fed import SnowyKITTI_FED
from tools.datasets.snowykitti_only import SnowyKITTI_ONLY
import os
import json
from federation.fedprox import Fedprox
import random
from tools.utils import set_seed


# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=10,
                    help='Number of clients. (default: 10)')
parser.add_argument('--active_rate', type=float, default=0.5,
                    help='active rate of clients in each round. (default: 0.5)')
parser.add_argument('--num_rounds', type=int, default=20,
                    help='Number of rounds. (default: 20)')
parser.add_argument('--num_epochs', type=int, default=2,
                    help='Number of epochs. (default: 2)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size in each training step. (default: 8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. (default: 1e-3)')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='Learning rate decay.')
parser.add_argument('--alpha', type=float, default=5.0)
parser.add_argument('--beta', type=float, default=0.5,
                    help='Relative weight of the FFT loss. Must be between 0 and 1.')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--dataset', type=str, default='both', choices=['snowykitti', 'wads', 'both'])
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--prox_mu', type=float, default=0.01)

# 是否使用分布引导的聚合机制
parser.add_argument('--agg_strategy', action='store_true')
parser.add_argument('--agg_factor', type=float, default=1.0)

parser.add_argument('--threshold', type=float, default=8e-3)
parser.add_argument('--z_ground', type=float, default=-1.8)
parser.add_argument('--snow_id', type=int, default=110)
parser.add_argument('--d_thresh', type=float, default=2.5)
parser.add_argument('--i_thresh', type=float, default=2/255)
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

# 保证参数的有效性
assert config['alpha'] > 1.0
assert 0 <= config['beta'] <= 1

# 学习率递减参数
if config['lr_decay'] < 0:
    config['lr_decay'] = 0.1**(1 / config['num_epochs'])

# client 自己的数据
data_loader_train = []
data_loader_val = []
ds_train = []
ds_val = []

if config['dataset'] == 'both':
    for client_id in range(0, 4):
        ds_train.append(WADS_FED('/home/fangzhy9/fed-desnow/data/wads', training=True, client_id=client_id))
        ds_val.append(WADS_FED('/home/fangzhy9/fed-desnow/data/wads', training=False, client_id=client_id))
    for client_id in range(4, config['num_clients']):
        ds_train.append(SnowyKITTI_FED('/home/fangzhy9/fed-desnow/data/snowyKITTI', training=True, client_id=client_id))
        ds_val.append(SnowyKITTI_FED('/home/fangzhy9/fed-desnow/data/snowyKITTI', training=False, client_id=client_id))
elif config['dataset'] == 'snowykitti':
    for client_id in range(0, config['num_clients']):
        ds_train.append(SnowyKITTI_ONLY('/home/fangzhy9/fed-desnow/data/snowyKITTI', training=True, client_id=client_id))
        ds_val.append(SnowyKITTI_ONLY('/home/fangzhy9/fed-desnow/data/snowyKITTI', training=False, client_id=client_id))
elif config['dataset'] == 'wads':
    for client_id in range(0, config['num_clients']):
        ds_train.append(WADS_ONLY('/home/fangzhy9/fed-desnow/data/wads', training=True, client_id=client_id))
        ds_val.append(WADS_ONLY('/home/fangzhy9/fed-desnow/data/wads', training=False, client_id=client_id))

# 划分数据集到每个client中
for i in range(config['num_clients']):
    data_loader_train.append( DataLoader(
        ds_train[i],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    ))
    data_loader_val.append( DataLoader(
        ds_val[i],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    ))

# active rate == 0.1, 0.2, 0.3时每一轮参与训练的client id
# 随着轮次变化，交替使用两个不同数据集的client参与训练，使客户端漂移的现象更明显，从而突出分布引导的聚合机制的作用
active_idx_list = { 
    # active 0.1
    1: [
        [0, ], [4, ], [1, ], [5, ], [2, ], [6, ], [3, ], [7, ], [8, ], [9, ],
        [0, ], [4, ], [1, ], [5, ], [2, ], [6, ], [3, ], [7, ], [8, ], [9, ],
    ],
    # active 0.2
    2: [
        [4, 5], [0, 1], [6, 7], [2, 3], [8, 9],
        [4, 5], [0, 1], [6, 7], [2, 3], [8, 9],
        [4, 5], [0, 1], [6, 7], [2, 3], [8, 9],
        [4, 5], [0, 1], [6, 7], [2, 3], [8, 9],
    ],
    # active 0.3
    3: [
        [0, 1, 2], [4, 5, 6], [3, 0, 1], [7, 8, 9],
        [2, 3, 0], [4, 5, 6], [1, 2, 3], [7, 8, 9],
        [0, 1, 2], [4, 5, 6], [3, 0, 1], [7, 8, 9],
        [2, 3, 0], [4, 5, 6], [1, 2, 3], [7, 8, 9],
        [0, 1, 2], [4, 5, 6], [3, 0, 1], [7, 8, 9],
    ]
}

# Fedprox方法
federation = Fedprox(config, device, data_loader_train, data_loader_val)

# 全部的clients下标
all_idx = list(range(config['num_clients']))

# 分发全局模型的参数到所有clients
federation.distribute(all_idx)

# 按设定轮次 进行训练
for r in range(config['num_rounds']):
    print(f"\n ***********  Federated training round {r+1}  *********** ")
    
    # 设置随机数生成器的种子
    random.seed(666 + r*r)

    # 随机选取本轮的参与clients个数k
    # k = random.randint(3, 8)
    k = round(config['num_clients']* config['active_rate'])
    print(f"Number of Clients to be trained this round: {k}")

    # 从0-9的下标中随机抽取k个数
    # active_idx = random.sample(range(config['num_clients']), k)
    # active_idx = [0, 1, 2, 3]
    if k >= 1 and k <= 3:
        active_idx = active_idx_list[k][r]
    else:
        active_idx = random.sample(range(config['num_clients']), k)

    print(f"Clients to be trained this round: {active_idx}")
    
    # 训练clients模型
    federation.train_clients(config, active_idx, device, r)
    
    # clients模型聚合到全局模型
    federation.aggregation(active_idx, agg_strategy=config['agg_strategy'], agg_factor=config['agg_factor'])
    
    # 保存全局模型
    federation.save_global_model(r)
    
    # 分发全局模型的参数到所有clients
    federation.distribute(all_idx)
    
    # 测试这一轮结束后每个client的本地模型在各自训练集上的性能
    federation.clients.test_clients_model(all_idx, r, config, device)
    
    # 保存clients模型
    federation.save_clients_model(r)
