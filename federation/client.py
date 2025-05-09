import os
from glob import glob
import torch
from torch.utils.data import DataLoader
import copy
from eval_model import eval_model
import json
import numpy as np

class Client():
    def __init__(self, config, data_loader_train: DataLoader, data_loader_val: DataLoader, model: torch.nn.Module, train_func, index: int, include_personaliezd: bool) -> None:
        # client的下标
        self.idx = index
        
        # client 自己的数据
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        # 训练集数据长度（帧数）
        self.data_len = len(data_loader_train.dataset)
        
        # client 自己的模型
        self.model = model
        
        # client 本地特征
        self.local_feature = None
        
        # client 自己的log 目录
        self.log_dir = os.path.join(config['log_dir'], config['tag'], "clients", str(self.idx))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 查看checkpoint 
        checkpoints = sorted(glob(os.path.join(self.log_dir, '*.pth')))
        if len(checkpoints):
            ckpt = checkpoints[-1]
            print(f'Loading the last checkpoint {ckpt:s}')
            self.model.load_state_dict(torch.load(ckpt))

            self.start_epoch = int(os.path.basename(ckpt).split('.')[0])
        else:
            self.start_epoch = 0
        
        # client 自己的优化器
        # 初始的学习率
        lr0 = config['lr'] * config['lr_decay']**self.start_epoch
        
        # 标记该client 是否包含 个性化模块
        self.include_personaliezd = include_personaliezd
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config['lr_decay'])
        '''
        if not include_personaliezd:
            # 没有个性化模块, 只需要一个优化器
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr0)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, config['lr_decay'])
        else:
            # 含有个性化模块, 需要分多个优化器
            # 个性化模块前的 公有模块
            self.optimizer_public = torch.optim.Adam(
                params=list(self.model.enc0.parameters())+
                list(self.model.enc1.parameters())+
                list(self.model.enc2.parameters())+
                list(self.model.enc3.parameters())+
                list(self.model.enc4.parameters())+
                list(self.model.dec4.parameters())+
                list(self.model.dec3.parameters())+
                list(self.model.dec2.parameters())+
                list(self.model.dec1.parameters())+
                list(self.model.dec0.parameters())+
                list(self.model.last_conv.parameters()),
                lr=lr0)
            self.scheduler_public = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_public, config['lr_decay'])
            
            # # 个性化模块后的 公有模块
            # self.optimizer_back = torch.optim.Adam([
            #     {'params': self.model.dec4.parameters()},
            #     {'params': self.model.dec3.parameters()},
            #     {'params': self.model.dec2.parameters()},
            #     {'params': self.model.dec1.parameters()},
            #     {'params': self.model.dec0.parameters()}
            #     ], lr=lr0)
            # self.scheduler_back = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_back, config['lr_decay'])
            
            # 个性化模块
            self.optimizer_personalized = torch.optim.Adam(list(self.model.selfatt.parameters()), lr=lr0)
            self.scheduler_personalized = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_personalized, config['lr_decay'])
            '''
        
        # 训练模式（方法），根据联邦学习方法选择，如FedAvg, FedProx
        self.train_func = train_func
        
    def train(self, config, device, fed_round, clients_count, active_clients_len, global_feature = None, mmd=None, mmd_nobn=False):
        # 按照设定的训练函数进行训练，训练函数由联邦学习的方法决定，如FedAvg, FedProx
        '''
        if self.include_personaliezd:
            self.local_feature = self.train_func(config, self.model, self.start_epoch, self.data_loader_train, self.data_loader_val,
                            self.optimizer_public, self.optimizer_personalized, self.log_dir,
                            self.scheduler_public, self.scheduler_personalized, device, 
                            fed_round, clients_count, active_clients_len, global_feature)
        else:
            self.local_feature = self.train_func(config, self.model, self.start_epoch, self.data_loader_train, self.data_loader_val, self.optimizer, 
                            self.log_dir, self.scheduler, device, fed_round, clients_count, active_clients_len, global_feature)
        '''
        if mmd==None:
            self.local_feature, loss_round = self.train_func(config, self.model, self.start_epoch, self.data_loader_train, self.data_loader_val, self.optimizer, 
                            self.log_dir, self.scheduler, device, fed_round, clients_count, active_clients_len, global_feature)
        else:
            print("mmd: ", mmd)
            self.local_feature, loss_round = self.train_func(config, self.model, self.start_epoch, self.data_loader_train, self.data_loader_val, self.optimizer, 
                            self.log_dir, self.scheduler, device, fed_round, clients_count, active_clients_len, global_feature, mmd, mmd_nobn)
        self.start_epoch += config['num_epochs']
        return loss_round
        
    def get_weights(self) -> dict:
        # 返回该client的模型参数
        return self.model.state_dict()
    
    def set_weights(self, weights: dict, nobn = False):
        # 根据输入的模型参数weights，设置该clients的模型参数
        '''
        # client存在个性化模块时（client的结构比server多一个个性化模块），需要server对应client的模块更新参数，全局模型分发参数不影响client的个性化模块
        '''
        # if self.include_personaliezd:
        client_weights = copy.deepcopy(self.model.state_dict())
        for key in weights.keys():
            if nobn:
                if 'bn' not in key:
                    client_weights[key] = weights[key]
            else:
                client_weights[key] = weights[key]
        self.model.load_state_dict(client_weights)
        # else:
        #     self.model.load_state_dict(weights)
        # 如果client的结构与server完全一致, 等价于 self.model.load_state_dict(weights)
    
    def get_local_feature(self):
        return self.local_feature
    
    def save_model(self, fed_round: int):
        fn_ckpt = os.path.join(self.log_dir, f'round_{(fed_round + 1):04d}.pth')
        print(f'\nSaving {fn_ckpt:s} ...')
        torch.save(self.get_weights(), fn_ckpt)
        
    def test_model(self, fed_round: int, config, device):
        # 在训练集上测试该client的本地模型，得到training metrics
        result_dict = eval_model(self.model, self.data_loader_train, config, device)
        with open(os.path.join(self.log_dir, f'metrics_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        return result_dict
    
        
# Client 组合
class ClientsAll():
    def __init__(self, config, data_loader_train: list[DataLoader], data_loader_val: list, model_class, device, train_func, include_personaliezd: bool) -> None:
        # 依次创建client，并返回全部clients的list
        self.clients = [Client(config, data_loader_train[i], data_loader_val[i], model_class().to(device), train_func, i, include_personaliezd) for i in range(config['num_clients'])]
        # 训练集的总帧数
        self.all_data_len = sum([self.clients[i].data_len for i in range(len(self.clients))])
        self.log_dir = os.path.join(config['log_dir'], config['tag'])
        
    def train(self, config, active_index: list, device, fed_round, global_feature = None, mmd=None, mmd_nobn=False):
        # 根据id选定一部分clients训练
        # 每一个client训练时打印 [训练的client下标]
        active_index_len = len(active_index)
        loss_clients = []
        for clients_count, idx in enumerate(active_index):
            print(' '.join([        
                f"training clients[{(idx)}] ...",
            ]))
            loss_clients.append(self.clients[idx].train(config, device, fed_round, clients_count, active_index_len, global_feature, mmd, mmd_nobn))
        
        loss_round = {}
        for key in loss_clients[0].keys():
            loss_round[key] = np.mean([loss[key] for loss in loss_clients])
        with open(os.path.join(self.log_dir, f'loss_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
            json.dump(loss_round, f, ensure_ascii=False, indent=4)
            
    def get_weights(self, clients_index: list) -> list:
        # 根据id返回对应的clients的模型参数list 
        return [self.clients[idx].get_weights() for idx in clients_index]
    
    def set_weights(self, weights, clients_index: list, nobn = False):
        # 根据id设置对应client的模型参数
        for idx in clients_index:
            self.clients[idx].set_weights(weights, nobn)
            
    def get_local_features(self, clients_index: list):
        # 根据id返回对应的clients的本地特征
        return [self.clients[idx].get_local_feature() for idx in clients_index]
    
    def get_data_len(self, clients_index: list):
        # 根据id返回对应的clients的训练集帧数
        return [self.clients[idx].data_len for idx in clients_index]
    
    def save_model(self, fed_round: int):
        for client in self.clients:
            client.save_model(fed_round)
            
    def test_clients_model(self, clients_index: list, fed_round: int, config, device):
        metrics_clients = []
        for idx in clients_index:
            metrics_clients.append(self.clients[idx].test_model(fed_round, config, device))
        metrics_round = {}
        for key in metrics_clients[0].keys():
            metrics_round[key] = np.mean([metrics[key] for metrics in metrics_clients])
        with open(os.path.join(self.log_dir, f'metrics_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_round, f, ensure_ascii=False, indent=4)