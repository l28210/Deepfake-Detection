from .server import Server
from .client import ClientsAll
from tools.models import LiSnowNet
# 不同的联邦学习方法主要的区别是 训练过程 和 损失函数, 因此针对不同的联邦学习方法写不同的训练函数
from .train_func.fedprox_train_func import train_fedprox
import copy
import torch

from .feder_utils import cos_similarity
from .feder_utils import count2proportion

class Fedprox():
    def __init__(self, config, device, data_loader_train, data_loader_val, global_data_loader_val=None) -> None:
        # 建立中心服务器
        self.server = Server(config, LiSnowNet().to(device), global_data_loader_val)
        # 建立多个客户端
        '''
            不同的联邦学习方法主要就是修改倒数的2个参数
            倒数第2个参数是训练方法(包括计算损失和更新模型权重)
            倒数第1个参数是 client模型是否包含 个性化模块
        '''
        self.clients = ClientsAll(config, data_loader_train, data_loader_val, LiSnowNet, device, train_fedprox, False)
        # 目前训练每个client的次数
        self.clients_count = [0 for _ in range(config['num_clients'])]
    
    def distribute(self, active_index: list):
        # 分发全局模型的参数到指定的（激活的）clients中，所有全局模型有的key都分发
        self.clients.set_weights(self.server.model.state_dict(), active_index, nobn = False)
    
    def train_clients(self, config, active_index, device, fed_round):
        self.clients.train(config, active_index, device, fed_round)    
    
    def aggregation(self, clients_index: list, agg_strategy=False, agg_factor=1.0):
        # 聚合指定的（激活的）clients的模型参数
        global_weights = copy.deepcopy(self.server.model.state_dict())
        all_clients_weights = self.clients.get_weights(clients_index)
        all_clients_data_len = self.clients.get_data_len(clients_index)
        data_len_sum = sum(all_clients_data_len)
        
        # 所有global有的key都聚合, 根据每个client的数据量按比例加权
        for key in global_weights.keys():
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
        
        # 更新每个client对全局模型的贡献比例
        for x in clients_index:
            self.clients_count[x] += 1
        
    def save_global_model(self, fed_round: int):
        # 保存聚合后的全局模型
        self.server.save_model(fed_round)
        
    def save_clients_model(self, fed_round: int):
        self.clients.save_model(fed_round)
