import torch
from collections import defaultdict
from .server import Server
from .client import ClientsAll
from tools.models import LiSnowNet
import copy
from .train_func.fednorm_train_func import train_fednorm
from .train_func.fednorm_train_func import grad_clients

from .feder_utils import cos_similarity
from .feder_utils import count2proportion


class Fednorm():
    def __init__(self, config, device, data_loader_train, data_loader_val, global_data_loader_val=None) -> None:
        # 建立中心服务器
        self.server = Server(config, LiSnowNet().to(device), global_data_loader_val)
        # 建立多个客户端
        self.clients = ClientsAll(config, data_loader_train, data_loader_val, LiSnowNet, device, train_fednorm, False)
        # 目前训练每个client的次数
        self.clients_count = [0 for _ in range(config['num_clients'])]
        self.config=config

    def distribute(self,active_index:list):
        # 分发全局模型的参数到指定的（激活的）clients中，所有全局模型有的key都分发
        self.clients.set_weights(self.server.model.state_dict(), active_index, nobn = False)

    def train_clients(self, config, active_index, device, fed_round):
        # 每轮训练开始时清空梯度列表
        grad_clients.clear()
        self.clients.train(config, active_index, device, fed_round) 
    
    def aggregation(self, clients_index: list,agg_strategy=False, agg_factor=1.0):
        # # 聚合指定的（激活的）clients的模型参数
        # global_weights = copy.deepcopy(self.server.model.state_dict())
        # all_clients_weights = self.clients.get_weights(clients_index)
        # all_clients_data_len = self.clients.get_data_len(clients_index)
        # data_len_sum = sum(all_clients_data_len)
        
        # # 所有global有的key都聚合, 根据每个client的数据量按比例加权
        # for key in global_weights.keys():
        #     global_weights[key] = torch.stack([client_weight[key].float() * (client_data_len / data_len_sum) for client_weight, client_data_len in zip(all_clients_weights, all_clients_data_len)]).sum(dim=0)
        
        # if agg_strategy:
        #     # 上一轮全局模型参数
        #     last_round_global_weight = copy.deepcopy(self.server.model.state_dict())
        #     cur_round_count = [1 if i in clients_index else 0 for i in range(len(self.clients_count))]
        #     # 计算聚合权重w，w越大，上一轮参数占比越大
        #     if sum(self.clients_count)==0:
        #         w = 0.0
        #     else:    
        #         w = agg_factor * 0.5 * (1.0 - cos_similarity(count2proportion(self.clients_count), count2proportion(cur_round_count)))
        #     for key in global_weights.keys():
        #         global_weights[key] = w * last_round_global_weight[key] + (1.0 - w) * global_weights[key]
        
        # self.server.model.load_state_dict(global_weights)


        # fednorm        
        # 累加所有client的梯度
        grad_all=defaultdict(lambda: None)
        for grad_each_client in grad_clients:
            for name,grad in grad_each_client.items():
                if grad_all[name] is None:
                    grad_all[name]=torch.zeros_like(grad)
                grad_all[name]+=grad
        # 对梯度进行正则化
        normalized_grad_all = {}
        for name,grad in grad_all.items():
            # 均值
            mean=grad.mean()
            # 标准差
            std=grad.std()
            # 每层中的梯度总数
            gama=grad.numel()
            normalized_grad_all[name]=(grad-mean)/((std+1e-8)*torch.sqrt(torch.tensor(gama, dtype=grad.dtype)))
            
        # 把参数传给一个client用于更新参数
        # for name,param in self.clients[0].model.named_parameters():
        #     if name in grad_all:
        #         param.grad=grad_all[name]
        for name, param in self.clients.clients[0].model.named_parameters():
            if name in normalized_grad_all:
                param.grad=normalized_grad_all[name]
        
        # 更新参数
        self.clients.clients[0].optimizer.step()
        # 将这个client参数移到server
        weight=copy.deepcopy(self.clients.clients[0].model.state_dict())
        
        # DGA聚合
        if agg_strategy:
            # server此时的参数还未更新，只有选定的client的参数是新的
            last_round_global_weight=copy.deepcopy(self.server.model.state_dict())
            this_round_global_weight=copy.deepcopy(self.clients.clients[0].model.state_dict())
            cur_round_count = [1 if i in clients_index else 0 for i in range(len(self.clients_count))]
            if sum(self.clients_count)==0:
                w=0.0
            else:
                w = agg_factor * 0.5 * (1.0 - cos_similarity(count2proportion(self.clients_count), count2proportion(cur_round_count)))
            for key in this_round_global_weight.keys():
                this_round_global_weight[key]=w*last_round_global_weight[key]+(1.0-w)*this_round_global_weight[key]
            weight=this_round_global_weight
            
        self.server.model.load_state_dict(weight)

    def save_global_model(self, fed_round: int):
        # 保存聚合后的全局模型
        self.server.save_model(fed_round)
        
    def save_clients_model(self, fed_round: int):
        self.clients.save_model(fed_round)