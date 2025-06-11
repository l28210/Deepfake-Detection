import torch
import numpy as np
import os
import time
from collections import defaultdict
from tools import losses
from torch.utils.tensorboard import SummaryWriter
import json
import copy

# # 获取前一轮的server的参数
global_server_weight=None

# 动态正则化系数列表
alpha_list=None
# 当前client对应下标
idx_alpha=None

def train_fedDyn(config, model, start_epoch, data_loader_train, data_loader_val, optimizer, log_dir, scheduler, device, fed_round, clients_count, active_clients_len, _):
    w1 = (2**config['alpha'] - 1) / 2**config['alpha']
    writer = SummaryWriter(log_dir)
    loss_epochs = []
    
    # 设置上一次迭代后的梯度
    # prev_client_grad={}
    # for name,param in model.state_dict().items():
    #     prev_client_grad[name]=torch.zeros_like(param)
    
    t_start = time.time()
    for epoch in range(start_epoch, start_epoch+config['num_epochs']):
        tb_train = defaultdict(list)
        model.train()
        for index, (fid, x, _, _) in enumerate(data_loader_train):
            x = x.to(device)

            # Forward
            idx_valid, y, _ = model(x)
            l_dwt, l_fft = losses.sparsity_loss(y)

            residual = (y - x) * idx_valid
            l_res = residual.abs().sum() / idx_valid.sum()

            l_sp = config['beta'] * l_dwt + (1 - config['beta']) * l_fft
            loss = w1 * l_sp + (1 - w1) * l_res
            
            # 获取当前模型参数
            current_model_weight=model.state_dict()
            
            # 计算上一次迭代后加入正则化项的梯度与client参数的内积
            # 由于计算的内积过大导致问题
            '''
            {
            "loss/train/DWT": 0.3757026906570663,
            "loss/train/FFT": 0.21686415163719136,
            "loss/train/Residual": 0.500233149398928,
            "loss/train/lsp": 0.29628342017531395,
            "loss/train/loss": 0.3007901064727617,
            "loss/train/loss_dot": 85283.12201922873,
            "loss/train/loss_norm": 1536.4253181789231,
            "loss/train/total_loss": -83746.39491736371
            }
            '''
            # loss_dot=0.0
            # for name,param in current_model_weight.items():
            #     # loss_dot += (param * prev_client_grad[name]).sum().item()
            #     loss_dot += (prev_client_grad[name]).sum().item()
                
                    
            # 计算正则化项
            # loss_norm=0.0
            # if global_server_weight is not None:
            #     for name,param in current_model_weight.items():
            #         loss_norm += (param - global_server_weight[name]-prev_client_grad[name]).pow(2).sum().item()
            # loss_norm*=config['alpha_coef']/2
            
            loss_algo=0.0
            for name,param in current_model_weight.items():
                loss_algo+=(config['alpha_coef']/alpha_list[idx_alpha])*torch.sum(param*(-global_server_weight[name]+param))
                
            # 计算总损失函数
            # 损失函数不可以为负数
            # total_loss=loss-loss_dot+loss_norm
            total_loss=loss+loss_algo
            # print(total_loss)
            # print(f'loss:{type(loss)}\nloss_algo{type(loss_algo)}\ntotal_loss:{type(total_loss)}')
            # print(loss.dtype)
            # print(loss_algo.dtype)
            
            
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 递归计算client梯度
            # new_client_weight=model.state_dict()
            # for name, param in prev_client_grad.items():
            #     prev_client_grad[name] = param - config['alpha_coef'] * (new_client_weight[name] - global_server_weight[name])
                
            # elapsed time
            mm, ss = divmod(time.time() - t_start, 60)
            hh, mm = divmod(mm, 60)

            print(' '.join([
                f"rounds: [{(fed_round + 1):4d}/{config['num_rounds']:4d}]",
                f"clients: [{(clients_count + 1):4d}/{active_clients_len:4d}]",
                f"epochs: [{(epoch + 1):4d}/{(start_epoch+config['num_epochs']):4d}]",
                f'batches: [{(index + 1):4d}/{len(data_loader_train):4d}]',
                f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
                f'losses: {l_dwt.item():.6f} {l_fft.item():.6f} {l_res.item():.6f} {loss_algo:.6f} {total_loss:.6f}'
            ]), end='\t\r')

            # tb_train['loss/train/DWT'].append(l_dwt.item())
            # tb_train['loss/train/FFT'].append(l_fft.item())
            # tb_train['loss/train/Residual'].append(l_res.item())
            # tb_train['loss/train/lsp'].append(l_sp.item())
            # tb_train['loss/train/loss'].append(loss.item())
            # tb_train['loss/train/loss_algo'].append(loss_algo)
            # tb_train['loss/train/total_loss'].append(total_loss.item())
            tb_train['loss/train/DWT'].append(l_dwt.cpu().item())
            tb_train['loss/train/FFT'].append(l_fft.cpu().item())
            tb_train['loss/train/Residual'].append(l_res.cpu().item())
            tb_train['loss/train/lsp'].append(l_sp.cpu().item())
            tb_train['loss/train/loss'].append(loss.cpu().item())
            tb_train['loss/train/loss_algo'].append(loss_algo.cpu().item())
            tb_train['loss/train/total_loss'].append(total_loss.cpu().item())

        loss_epoch = {}
        for key, value in tb_train.items():
            loss_epoch[key] = np.nanmean(value)
            writer.add_scalar(key, np.nanmean(value), epoch + 1)
        loss_epochs.append(loss_epoch)

        writer.flush()
        scheduler.step()

    writer.close()
    
    # for name,param in model.state_dict().items():
    #     print(f'{name}:{param}')
    
    # 保存 client 在这一个round 在训练集上的loss
    loss_round = {}
    for key in loss_epochs[0].keys():
        loss_round[key] = np.mean([l_epoch[key] for l_epoch in loss_epochs])
    with open(os.path.join(log_dir, f'loss_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
        json.dump(loss_round, f, ensure_ascii=False, indent=4)
    
    return None, loss_round