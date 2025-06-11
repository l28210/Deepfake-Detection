from .server import Server
from .client import ClientsAll
from tools.models import LiSnowNet
from tools.models_att import LiSnowNetAtt
# 不同的联邦学习方法主要的区别是 训练过程 和 损失函数, 因此针对不同的联邦学习方法写不同的训练函数
from .train_func.fedmine_train_func import train_fedmine
import copy
import torch
import numpy as np

# 两个分布的余弦相似度
def cos_similarity(distribution_a:list, distribution_b:list):
    a = np.array(distribution_a)
    b = np.array(distribution_b)
    return np.dot(a,b)/(np.linalg.norm(a)*(np.linalg.norm(b)))

# 每个clients的个数 转换 分布比例
def count2proportion(cli_count:list) -> list:
    total_count = sum(cli_count)
    return [x/total_count for x in cli_count]

class Fedmine():
    def __init__(self, config, device, data_loader_train, data_loader_val, global_feature, global_data_loader_val=None, personaliezd=True, mmd=True, mmd_nobn=False) -> None:
        # 建立中心服务器
        self.server = Server(config, LiSnowNet().to(device), global_data_loader_val)
        # 建立多个客户端
        '''
            不同的联邦学习方法主要就是修改倒数的2个参数
            倒数第2个参数是训练方法(包括计算损失和更新模型权重)
            倒数第1个参数是 client模型是否包含 个性化模块
        '''
        self.mmd = mmd
        self.mmd_nobn = mmd_nobn
        if personaliezd:
            self.clients = ClientsAll(config, data_loader_train, data_loader_val, LiSnowNetAtt, device, train_fedmine, include_personaliezd=True)
        else:
            self.clients = ClientsAll(config, data_loader_train, data_loader_val, LiSnowNet, device, train_fedmine, include_personaliezd=False)
        # 全局特征, 用于结合 本地特征 计算MMD loss
        self.global_feature = global_feature

        # 目前训练每个client的次数
        self.clients_count = [0 for _ in range(config['num_clients'])]

    def distribute(self, active_index: list, nobn=False):
        # 分发全局模型的参数到指定的（激活的）clients中，默认所有全局模型有的key都分发，nobn==True则不分发BN层参数
        self.clients.set_weights(self.server.model.state_dict(), active_index, nobn)
    
    def train_clients(self, config, active_index, device, fed_round):
        # for x in active_index:
        #     self.clients_count[x] += 1
        self.clients.train(config, active_index, device, fed_round, self.global_feature, self.mmd, mmd_nobn=self.mmd_nobn)
    
    def aggregation(self, clients_index: list, nobn = False, glob_fea_update_strategy=False, omega_factor=1.0, agg_strategy=False, agg_factor=1.0):
        # 聚合指定的（激活的）clients的模型参数
        global_weights = copy.deepcopy(self.server.model.state_dict())
        all_clients_weights = self.clients.get_weights(clients_index)
        all_clients_data_len = self.clients.get_data_len(clients_index)
        data_len_sum = sum(all_clients_data_len)
        
        # 所有global有的key都聚合, 根据每个client的数据量按比例加权
        for key in global_weights.keys():
            if nobn:
                if 'bn' not in key:
                    global_weights[key] = torch.stack([client_weight[key].float() * (client_data_len / data_len_sum) for client_weight, client_data_len in zip(all_clients_weights, all_clients_data_len)]).sum(dim=0)
            else:
                global_weights[key] = torch.stack([client_weight[key].float() * (client_data_len / data_len_sum) for client_weight, client_data_len in zip(all_clients_weights, all_clients_data_len)]).sum(dim=0)
        if agg_strategy:
            last_round_global_weight = copy.deepcopy(self.server.model.state_dict())
            cur_round_count = [1 if i in clients_index else 0 for i in range(len(self.clients_count))]
            if sum(self.clients_count)==0:
                w = 0.0
            else:    
                w = agg_factor * 0.5 * (1.0 - cos_similarity(count2proportion(self.clients_count), count2proportion(cur_round_count)))
            for key in global_weights.keys():
                global_weights[key] = w * last_round_global_weight[key] + (1.0 - w) * global_weights[key]
        self.server.model.load_state_dict(global_weights)
        
        # 更新全局特征
        clients_features = self.clients.get_local_features(clients_index)
        global_feature_prime = torch.stack(clients_features, dim=0).mean(dim=0)
        if glob_fea_update_strategy and self.global_feature != None:
            cur_round_count = [1 if i in clients_index else 0 for i in range(len(self.clients_count))]
            omega = omega_factor * 0.5 * (1.0 - cos_similarity(count2proportion(self.clients_count), count2proportion(cur_round_count)))
            self.global_feature = omega * self.global_feature + (1.0 - omega) * global_feature_prime
        else:
            self.global_feature = global_feature_prime
            
        for x in clients_index:
            self.clients_count[x] += 1
        
    def save_global_model(self, fed_round: int):
        # 保存聚合后的全局模型
        self.server.save_model(fed_round)
        
    def save_clients_model(self, fed_round: int):
        self.clients.save_model(fed_round)
