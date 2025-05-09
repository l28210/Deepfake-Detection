import torch
import torch.nn as nn
from glob import glob
import os

# Server
class Server():
    def __init__(self, config, global_model: nn.Module, data_loader_val = None) -> None:
        # server的全局模型
        self.model = global_model
        
        # server 自己的log 目录
        self.log_dir = os.path.join(config['log_dir'], config['tag'], "server")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 查看checkpoint 
        checkpoints = sorted(glob(os.path.join(self.log_dir, '*.pth')))
        if len(checkpoints):
            ckpt = checkpoints[-1]
            print(f'Loading the last checkpoint {ckpt:s}')
            self.model.load_state_dict(torch.load(ckpt))
        
        # server的val数据集，就是全部clients的val数据集的并集，目前没用上，默认是None
        self.data_loader_val = data_loader_val
    
    def save_model(self, fed_round):
        # 保存全局模型
        fn_ckpt = os.path.join(self.log_dir, f'{(fed_round + 1):04d}.pth')
        print(f'Saving {fn_ckpt:s} ...')
        torch.save(self.model.state_dict(), fn_ckpt)
