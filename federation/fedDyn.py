from .server import Server
from .client import ClientsAll
from tools.models import LiSnowNet
from .train_func.fedDyn_train_func import train_fedDyn
from .train_func import fedDyn_train_func
import copy
import torch

from .feder_utils import cos_similarity
from .feder_utils import count2proportion



class FedDyn():
    def __init__(self, config, device, data_loader_train, data_loader_val, global_data_loader_val=None) -> None:
        # 建立中心服务器
        self.server = Server(config, LiSnowNet().to(device), global_data_loader_val)
        # 建立多个客户端
        '''
            不同的联邦学习方法主要就是修改倒数的2个参数
            倒数第2个参数是训练方法(包括计算损失和更新模型权重)
            倒数第1个参数是 client模型是否包含 个性化模块
        '''
        self.clients = ClientsAll(config, data_loader_train, data_loader_val, LiSnowNet, device, train_fedDyn, False)
        # 目前训练每个client的次数
        self.clients_count = [0 for _ in range(config['num_clients'])]
        
        self.config=config
        
        # 初始化差异列表
        # self.diff_client_server = [defaultdict(lambda: {name: torch.zeros_like(param) for name, param in self.server.model.state_dict().items()}) for _ in data_loader_train]
        
        # self.diff_client_server=[]
        # for idx in data_loader_train:
        #     self.diff_client_server.append(defaultdict(lambda: {name: torch.zeros_like(param) for name, param in self.server.model.state_dict().items()}))
        self.diff_client_server = [
            {name: torch.zeros_like(param) for name, param in self.server.model.state_dict().items()}
            for _ in data_loader_train
        ]
        
        
    
    def distribute(self, active_index: list):
        # 分发全局模型的参数到指定的（激活的）clients中，所有全局模型有的key都分发
        self.clients.set_weights(self.server.model.state_dict(), active_index, nobn = False)
    
    def train_clients(self, config, active_index, device, fed_round):
        # 获取server参数
        fedDyn_train_func.global_server_weight=copy.deepcopy(self.server.model.state_dict())
        self.clients.train(config, active_index, device, fed_round)    
    
    def aggregation(self, clients_index: list, agg_strategy=False, agg_factor=1.0):
        # print(len(self.diff_client_server))   10
        # print(self.config['num_clients'])     10
        # 获取全局模型参数
        global_weights = copy.deepcopy(self.server.model.state_dict())
        fedDyn_train_func.global_server_weight=copy.deepcopy(self.server.model.state_dict())
        
        # 利用fedavg计算选取的clients的参数的均值
        all_clients_weights = self.clients.get_weights(clients_index)
        all_clients_data_len = self.clients.get_data_len(clients_index)
        data_len_sum = sum(all_clients_data_len)
        
        # 利用fedavg计算出全局参数更新第一项
        # 所有global有的key都聚合, 根据每个client的数据量按比例加权
        avg_clients_weight={name: torch.zeros_like(param) for name, param in global_weights.items()}
        for key in global_weights.keys():
            avg_clients_weight[key] = torch.stack([client_weight[key].float() * (client_data_len / data_len_sum) for client_weight, client_data_len in zip(all_clients_weights, all_clients_data_len)]).sum(dim=0)
        
        # print('fedavg')
        # for name,param in avg_clients_weight.items():
        #     print(f'{name}:{param}')
        '''
        fedavg
        enc0.0.main_block.0.conv.weight:tensor([[[[ 1.2137e-01, -4.4548e-03, -1.2740e-01],
          [-8.0333e-02,  7.6769e-02, -1.9370e-01],
          [ 1.2826e-01,  1.2216e-01,  7.1927e-02]],
        '''
        
        
        # print('weights of server:')
        # for name, param in sum_clients_weight.items():
        #     print(f'name:{name}\nparam:{param}')
        # 确定是由于全局参数更新导致的nan问题
        
        # 计算累计差异
        for idx in clients_index:
            client = self.clients.clients[idx]
            client_weight = copy.deepcopy(client.model.state_dict())
            for name, param in client_weight.items():
                # self.diff_client_server[idx][name]+=param-global_weights[name]
                self.diff_client_server[idx][name]+=(param - global_weights[name]).to(self.diff_client_server[idx][name].dtype)
        
        # 计算差异化均值
        mean_diff={name: torch.zeros_like(param,dtype=param.dtype) for name, param in avg_clients_weight.items()}
        for name,param in mean_diff.items():
            # diff_name = torch.stack([self.diff_client_server[idx][name].float() for idx in clients_index])
            # mean_diff[name]=torch.mean(diff_name,dim=0).to(param.dtype)
            
            mean_diff[name]=torch.stack([client_weight[name].float()/len(self.diff_client_server) for client_weight in self.diff_client_server]).sum(dim=0)
        
        # print('mean_diff')
        # for name,param in mean_diff.items():
        #     print(f'name:{name}\nparam:{param}')
        '''
        name:enc0.0.main_block.0.conv.weight
        param:tensor([[[[ 0.0019,  0.0016, -0.0033],
          [ 0.0037, -0.0032, -0.0013],
          [ 0.0042, -0.0005, -0.0039]],
        '''
        
        # 更新参数
        sum_clients_weight={name:torch.zeros_like(param) for name,param in global_weights.items()}
        for name,param in avg_clients_weight.items():
            sum_clients_weight[name]=(avg_clients_weight[name]+mean_diff[name]).to(dtype=torch.float32)

            if avg_clients_weight[name].dtype != mean_diff[name].dtype:
                print(f'type of avg{avg_clients_weight[name].dtype}')
                print(f'type of dyn{mean_diff[name].dtype}')
        # print('weights of server:')
        # for name, param in sum_clients_weight.items():
        #     print(f'name:{name}\nparam:{param}')
        '''
        name:enc0.0.main_block.0.conv.weight
        param:tensor([[[[ 1.3969e-01,  9.0046e-03, -1.6189e-01],
          [-4.3939e-02,  4.5778e-02, -2.0580e-01],
          [ 1.7283e-01,  1.1694e-01,  3.3493e-02]],
          '''
        
        # key='enc0.0.main_block.0.conv.weight'
        # print(f'fedavg:\n{avg_clients_weight[key][0][0]}')
        # print(f'mean_diff:\n{mean_diff[key][0][0]}')
        # print(f'sum_weight:\n{sum_clients_weight[key][0][0]}')

            
            
        # print('self.h:')
        # for name,param in self.h.items():
        #     print(f'name:{name}\nparam:{param}')
        
        # sum_clients_weight = defaultdict(lambda: {name: torch.zeros_like(param) for name, param in global_weights.items()})
        # for idx in clients_index:
        #     client = self.clients.clients[idx]
        #     client_weight = copy.deepcopy(client.model.state_dict())
        #     for name, param in client_weight.items():
        #         if name not in sum_clients_weight:
        #             sum_clients_weight[name] = param
        #         else:
        #             sum_clients_weight[name] += param
        
        # print('weights of server:')
        # for name, param in sum_clients_weight.items():
        #     print(f'name:{name}\nparam:{param}')
        
        # for name, param in sum_clients_weight.items():
        #     # sum_clients_weight[name] = (sum_clients_weight[name] / (self.config['num_clients'] * self.config['active_rate'])).to(global_weights[name].dtype)
        #     # sum_clients_weight[name] = torch.div(sum_clients_weight[name],(self.config['num_clients'] * self.config['active_rate']))
        #     # sum_clients_weight[name] -= (self.h[name] / self.config["alpha_coef"]).to(global_weights[name].dtype)
        #     # sum_clients_weight[name] -= torch.div(self.h[name],self.config["alpha_coef"])
        #     # sum_clients_weight[name] -= self.h[name]
            
        #     sum_clients_weight[name]-=self.h[name]
            
        # print('weights of server:')
        # for name, param in sum_clients_weight.items():
        #     print(f'name:{name}\nparam:{param}')
        
        
        
        # fedDyn跑的参数
        '''
        weights of server:
        name:enc0.0.main_block.0.conv.weight
        param:tensor([[[[ 0.1234, -0.0045, -0.1328],
          [-0.0772,  0.0758, -0.1943],
          [ 0.1321,  0.1225,  0.0686]],
        '''
        # fedavg
        '''
        weights of server:
        name:enc0.0.main_block.0.conv.weight
        param:tensor([[[[ 1.2088e-01, -2.1328e-03, -1.2836e-01],
          [-8.0508e-02,  7.7176e-02, -1.9381e-01],
          [ 1.2395e-01,  1.2309e-01,  7.2485e-02]],
        '''
        # 怀疑是计算过程精度导致问题
        # 确实是，直接用乘法不会导致精度降低，
        
        if agg_strategy:
            last_round_global_weight = global_weights
            this_round_global_weight = sum_clients_weight
            cur_round_count = [1 if i in clients_index else 0 for i in range(len(self.clients_count))]
            if sum(self.clients_count) == 0:
                w = 0.0
            else:
                w = agg_factor * 0.5 * (1.0 - cos_similarity(count2proportion(self.clients_count), count2proportion(cur_round_count)))
            for key in this_round_global_weight.keys():
                this_round_global_weight[key] = w * last_round_global_weight[key] + (1.0 - w) * this_round_global_weight[key]
            sum_clients_weight = this_round_global_weight
            
        # self.server.model.load_state_dict(sum_clients_weight)
        self.server.model.load_state_dict(avg_clients_weight)
        # self.server.model.load_state_dict(mean_diff)
        
        
        
        for x in clients_index:
            self.clients_count[x] += 1
        

                
    def save_global_model(self, fed_round: int):
        # 保存聚合后的全局模型
        self.server.save_model(fed_round)
        
    def save_clients_model(self, fed_round: int):
        self.clients.save_model(fed_round)
