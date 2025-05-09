from .server import Server
from .client import ClientsAll
from model.replknet import create_RepLKNet31B
# 不同的联邦学习方法主要的区别是 训练过程 和 损失函数, 因此针对不同的联邦学习方法写不同的训练函数
from .train_func.fedavg_train_func import train_fedavg

import copy
import torch

class Fedavg():
    def __init__(self, config, device, data_loader_train, data_loader_val, global_data_loader_val=None) -> None:
        # 建立中心服务器
        self.server = Server(config, create_RepLKNet31B(num_classes=2,use_checkpoint=True).to(device), global_data_loader_val)
        # 建立多个客户端
        '''
            不同的联邦学习方法主要就是修改倒数的2个参数
            倒数第2个参数是训练方法(包括计算损失和更新模型权重)
            倒数第1个参数是 client模型是否包含 个性化模块
        '''
        self.clients = ClientsAll(config, data_loader_train, data_loader_val, create_RepLKNet31B, device, train_fedavg, False)
        # 目前训练每个client的次数
        self.clients_count = [0 for _ in range(config['num_clients'])]