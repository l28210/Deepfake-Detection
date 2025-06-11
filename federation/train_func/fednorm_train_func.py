import os
import json
import time
import torch
from collections import defaultdict
from tools import losses
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 所有client训练完成后累加的梯度作为一个元素的列表
grad_clients=[]

def train_fednorm(config, model, start_epoch, data_loader_train, data_loader_val, optimizer, log_dir, scheduler, device, fed_round, clients_count, active_clients_len, _):
    # 超参数alpha设置权重
    w1=(2**config["alpha"]-1)/2**config['alpha']
    writer=SummaryWriter(log_dir)
    loss_epochs=[]
    
    # 获取一个client在这一轮训练累计的的梯度
    local_grad = defaultdict(lambda: None)
    

    # 记录时间戳
    t_start=time.time()
    # 对一个clients的每个epoch计算损失
    for epoch in range(start_epoch,start_epoch+config['num_epochs']):
        # defaultdict是dict的子类,当尝试访问一个不存在的键时,会自动创建一个空list
        tb_train=defaultdict(list)
        # 将模型设置为训练模式
        model.train()
        
        # 对单个client的每个batch数据遍历
        for index,(fid,x,_,_) in enumerate(data_loader_train):
            # 将张量x移动到指定的CPU或GPU
            x=x.to(device)

            # 前向传播
            idx_valid,y,_=model(x)
            l_dwt,l_fft=losses.sparsity_loss(y)

            residual = (y - x) * idx_valid
            l_res = residual.abs().sum() / idx_valid.sum()

            l_sp = config['beta'] * l_dwt + (1 - config['beta']) * l_fft
            loss = w1 * l_sp + (1 - w1) * l_res

            # 反向传播与优化,由于fednorm在client不更新参数,只是累加损失
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            
            # 测试获取梯度方法
            # loss.backward()
            # gradients={}
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         gradients[name] = param.grad.clone()
            # for name, grad in gradients.items():
            #     print(f"{name}: {grad}")
            # 可以准确获取梯度
            '''
            enc0.0.main_block.0.bn.weight: tensor([ 0.0018,  0.0030, -0.0013,  0.0023, -0.0075,  0.0057,  0.0002, -0.0042],
            device='cuda:0')
            enc0.0.main_block.0.bn.bias: tensor([ 0.0027,  0.0023, -0.0002,  0.0042, -0.0083,  0.0046, -0.0007, -0.0020],
            device='cuda:0')
            '''
            
            # 获取这个batch的梯度
            loss.backward()
            gradients={}
            for name,param in model.named_parameters():
                if param.grad is not None:
                    gradients[name]=param.grad.clone()
                    
            # 累加获得的梯度
            for name,grad in gradients.items():
                if local_grad[name] is None:
                    local_grad[name]=torch.zeros_like(grad)
                    local_grad[name]+=grad
            
            # 梯度清零
            optimizer.zero_grad()
            

            # 计算并格式化时间差
            mm, ss = divmod(time.time() - t_start, 60)
            hh, mm = divmod(mm, 60)

            print(' '.join([
                f"rounds: [{(fed_round + 1):4d}/{config['num_rounds']:4d}]",
                f"clients: [{(clients_count + 1):4d}/{active_clients_len:4d}]",
                f"epochs: [{(epoch + 1):4d}/{(start_epoch+config['num_epochs']):4d}]",
                f'batches: [{(index + 1):4d}/{len(data_loader_train):4d}]',
                f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
                f'losses: {l_dwt.item():.6f} {l_fft.item():.6f} {l_res.item():.6f}'
            ]), end='\t\r')

            tb_train['loss/train/DWT'].append(l_dwt.item())
            tb_train['loss/train/FFT'].append(l_fft.item())
            tb_train['loss/train/Residual'].append(l_res.item())
            tb_train['loss/train/lsp'].append(l_sp.item())
            tb_train['loss/train/loss'].append(loss.item())

        # 获取一个epoch中每种损失的值对应的字典
        loss_epoch = {}
        for key, value in tb_train.items():
            # key是损失函数的名称,value是当前训练周期的每个batch的损失函数值列表
            # np.nanmean(value)计算value列表的平均值,并忽略nan
            loss_epoch[key] = np.nanmean(value)
            # 计算每种损失的平均值
            writer.add_scalar(key, np.nanmean(value), epoch + 1)
        loss_epochs.append(loss_epoch)


        #  将所有积累的日志数据立即写入磁盘
        writer.flush()
        
        # 由于这里不更新参数，故不更新学习率调度器
        # 更新学习率调度器，根据预定义的策略调整优化器的学习率，以帮助模型更有效地收敛。
        # scheduler.step()

    writer.close()
    
    grad_clients.append(local_grad)

    # 保存 client 在这一个round 在训练集上的loss
    loss_round = {}
    for key in loss_epochs[0].keys():
        loss_round[key] = np.mean([l_epoch[key] for l_epoch in loss_epochs])
    with open(os.path.join(log_dir, f'loss_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
        json.dump(loss_round, f, ensure_ascii=False, indent=4)

    return None,loss_round